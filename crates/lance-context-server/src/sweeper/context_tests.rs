use std::sync::Arc;
use std::time::Duration;

use axum::{extract::State, Json};
use lance_context_api::{AddRecordRequest, CreateContextRequest};
use lance_context_core::{record_from_add_request, ContextStore, MergeMemoryBudget};
use tempfile::TempDir;
use tokio::sync::{RwLock, Semaphore};
use tokio::time::{sleep, timeout};

use super::{merge_pass_coordinated, Sweepable};
use crate::state::AppState;

const TEST_TIMEOUT: Duration = Duration::from_secs(30);

async fn create_context(state: &Arc<AppState>) -> Arc<RwLock<ContextStore>> {
    let (_, Json(created)) = crate::routes::contexts::create_context(
        State(state.clone()),
        Json(CreateContextRequest {
            name: "shared".into(),
            storage_options: None,
            id_index_type: None,
            blob_columns: None,
            embedding_dim: None,
            distance_metric: None,
        }),
    )
    .await
    .unwrap();
    assert_eq!(created.name, "shared");
    state.get_or_open_context_store("shared").await.unwrap()
}

async fn add(store: &Arc<RwLock<ContextStore>>, id: &str) {
    let record = record_from_add_request(
        &AddRecordRequest {
            role: "user".into(),
            content_type: "text/plain".into(),
            text_payload: Some(format!("payload-{id}")),
            ..Default::default()
        },
        id.into(),
        "run".into(),
    );
    store.read().await.add(&[record]).await.unwrap();
}

async fn pending(store: &Arc<RwLock<ContextStore>>) -> usize {
    store.read().await.pending_wal_generations().await.unwrap()
}

async fn assert_records(store: &Arc<RwLock<ContextStore>>, expected: &[&str]) {
    let records = store.read().await.list(None, None).await.unwrap();
    let mut actual: Vec<_> = records
        .into_iter()
        .map(|record| (record.id, record.text_payload.unwrap()))
        .collect();
    actual.sort();
    let mut expected: Vec<_> = expected
        .iter()
        .map(|id| (id.to_string(), format!("payload-{id}")))
        .collect();
    expected.sort();
    assert_eq!(actual, expected);
}

#[tokio::test]
async fn global_cleanup_timer_merges_context_wal_without_losing_records() {
    timeout(TEST_TIMEOUT, async {
        let dir = TempDir::new().unwrap();
        let mut state = AppState::new_for_test(dir.path().to_path_buf()).await;
        state.rollout_cleanup_interval_secs = 1;
        let state = Arc::new(state);
        let store = create_context(&state).await;
        add(&store, "first").await;
        add(&store, "second").await;
        assert_eq!(pending(&store).await, 2);

        let sweeper = state.spawn_global_sweeper().unwrap();
        while pending(&store).await != 0 {
            sleep(Duration::from_millis(20)).await;
        }
        sweeper.abort();

        assert_records(&store, &["first", "second"]).await;
    })
    .await
    .expect("context WAL was not merged by the global cleanup timer");
}

#[tokio::test]
async fn context_cleanup_is_bounded_and_only_merges_its_own_writer() {
    timeout(TEST_TIMEOUT, async {
        let dir = TempDir::new().unwrap();
        let mut state_a =
            AppState::new_for_test_with_instance(dir.path().to_path_buf(), Some("worker-a".into()))
                .await;
        state_a.rollout_merge_max_generations = 1;
        let state_a = Arc::new(state_a);
        let a = create_context(&state_a).await;
        add(&a, "a1").await;
        add(&a, "a2").await;

        let state_b = Arc::new(
            AppState::new_for_test_with_instance(dir.path().to_path_buf(), Some("worker-b".into()))
                .await,
        );
        let b = state_b.get_or_open_context_store("shared").await.unwrap();
        add(&b, "b1").await;
        add(&b, "b2").await;
        assert_eq!(pending(&a).await, 4);

        assert_eq!(a.merge_wal().await.unwrap(), 1);
        assert_eq!(pending(&a).await, 3);
        assert_eq!(a.merge_wal().await.unwrap(), 1);
        assert_eq!(pending(&a).await, 2);
        assert_eq!(a.merge_wal().await.unwrap(), 0);

        // A's cleanup must neither merge B's generations nor fence B's writer.
        add(&b, "b3").await;
        assert_eq!(pending(&b).await, 3);
        assert_eq!(b.merge_wal().await.unwrap(), 3);
        assert_eq!(pending(&a).await, 0);
        for store in [&a, &b] {
            assert_records(store, &["a1", "a2", "b1", "b2", "b3"]).await;
        }
    })
    .await
    .expect("bounded, owner-local context cleanup did not complete");
}

#[tokio::test]
async fn context_cleanup_skips_busy_shared_slots_and_retries_next_pass() {
    timeout(TEST_TIMEOUT, async {
        let dir = TempDir::new().unwrap();
        let mut state = AppState::new_for_test(dir.path().to_path_buf()).await;
        let slots = Arc::new(Semaphore::new(1));
        state.merge_slots = Some(slots.clone());
        let state = Arc::new(state);
        let store = create_context(&state).await;
        add(&store, "first").await;

        let held = slots.clone().acquire_owned().await.unwrap();
        timeout(
            Duration::from_secs(1),
            merge_pass_coordinated(
                vec![("shared".into(), store.clone())],
                TEST_TIMEOUT,
                Some(state.clone()),
            ),
        )
        .await
        .expect("cleanup must skip a busy slot instead of waiting");
        assert_eq!(pending(&store).await, 1);

        drop(held);
        merge_pass_coordinated(
            vec![("shared".into(), store.clone())],
            TEST_TIMEOUT,
            Some(state),
        )
        .await;
        assert_eq!(pending(&store).await, 0);
        assert_eq!(slots.available_permits(), 1);
        assert_records(&store, &["first"]).await;
    })
    .await
    .expect("context cleanup did not resume after a merge slot became free");
}

#[tokio::test]
async fn context_cleanup_waits_for_shared_memory_without_blocking_reads_or_appends() {
    timeout(TEST_TIMEOUT, async {
        let dir = TempDir::new().unwrap();
        let budget = MergeMemoryBudget::new(1024 * 1024);
        let mut state = AppState::new_for_test(dir.path().to_path_buf()).await;
        state.merge_budget = Some(budget.clone());
        state.rollout_merge_max_bytes = budget.limit();
        let state = Arc::new(state);
        let store = create_context(&state).await;
        add(&store, "first").await;

        let held = budget.reserve(budget.limit()).await;
        let merge = store.merge_wal();
        tokio::pin!(merge);
        assert!(
            timeout(Duration::from_millis(100), &mut merge)
                .await
                .is_err(),
            "cleanup must wait for the shared merge memory budget"
        );

        timeout(Duration::from_secs(5), async {
            assert_records(&store, &["first"]).await;
            add(&store, "during-prepare").await;
            assert_records(&store, &["first", "during-prepare"]).await;
        })
        .await
        .expect("waiting for merge memory must not hold the exclusive store lock");
        assert_eq!(pending(&store).await, 2);
        assert!(
            timeout(Duration::from_millis(100), &mut merge)
                .await
                .is_err(),
            "cleanup completed while the merge memory budget was occupied"
        );

        drop(held);
        let merged = merge.await.unwrap();
        assert!(merged > 0);
        assert_eq!(budget.reserved(), 0);
        assert_eq!(store.merge_wal().await.unwrap() + merged, 2);
        assert_eq!(pending(&store).await, 0);
        assert_eq!(budget.reserved(), 0);
        assert_records(&store, &["first", "during-prepare"]).await;
    })
    .await
    .expect("context cleanup did not complete after merge memory became free");
}
