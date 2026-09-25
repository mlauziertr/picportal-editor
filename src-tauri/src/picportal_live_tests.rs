//! Opt-in round trip against the production PicPortal API (MAX-22).
//!
//! Ignored by default. Run with a dedicated test account:
//! `PICPORTAL_LIVE_EMAIL=… PICPORTAL_LIVE_PASSWORD=… cargo test --lib live_tests -- --ignored --nocapture`
//!
//! It creates a draft gallery `test-editor-<timestamp>` without face filtering,
//! publishes ten generated JPEGs through `publish_one`, and prints `LIVE` lines
//! without credentials or cookie values. The gallery id is printed so the
//! caller can inspect and delete it afterwards; the test never deletes data.

use super::*;
use serde_json::json;

fn live(event: &str, value: serde_json::Value) {
    println!("LIVE {event} {value}");
}

fn cookie_names(session: &PicPortalSession) -> Vec<String> {
    session
        .cookies
        .lock()
        .map(|cookies| {
            cookies
                .iter()
                .filter_map(|cookie| cookie.split_once('=').map(|(name, _)| name.to_owned()))
                .collect()
        })
        .unwrap_or_default()
}

fn write_test_jpeg(directory: &Path, index: u32) -> String {
    let (width, height) = (1600, 1067);
    let image = image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([
            ((x * 255 / width + index * 23) % 256) as u8,
            ((y * 255 / height + index * 41) % 256) as u8,
            (((x / 80 + y / 80 + index) % 2) * 160 + 40) as u8,
        ])
    });
    let path = directory.join(format!("test-editor-{index:02}.jpg"));
    image.save(&path).expect("write test JPEG");
    path.to_string_lossy().into_owned()
}

struct LiveHarness<'a> {
    session: &'a PicPortalSession,
    gallery: &'a GallerySummary,
    capabilities: &'a ProcessingCapabilities,
    state: PicPortalState,
    queue_path: PathBuf,
    queue: QueueFile,
    cancel_before: Option<usize>,
    progress: Vec<(usize, usize)>,
    refresh_attempts: usize,
}

impl<'a> LiveHarness<'a> {
    fn new(
        session: &'a PicPortalSession,
        gallery: &'a GallerySummary,
        capabilities: &'a ProcessingCapabilities,
        queue_path: PathBuf,
    ) -> Self {
        let queue = load_queue(&queue_path).expect("queue");
        Self {
            session,
            gallery,
            capabilities,
            state: PicPortalState::default(),
            queue_path,
            queue,
            cancel_before: None,
            progress: Vec::new(),
            refresh_attempts: 0,
        }
    }

    async fn run(&mut self, paths: Vec<String>) -> PublicationOutcome {
        let mut refresh_available = true;
        let outcome = drive_publication(paths, &mut refresh_available, self).await;
        save_queue(&self.queue_path, &self.queue).expect("save queue");
        outcome
    }
}

impl PublicationDriver for LiveHarness<'_> {
    fn is_cancelled(&self) -> bool {
        self.state.publish_cancelled.load(Ordering::SeqCst)
    }

    fn progress(&mut self, current: usize, total: usize) {
        self.progress.push((current, total));
        if self.cancel_before == Some(current) {
            self.state.publish_cancelled.store(true, Ordering::SeqCst);
        }
    }

    async fn publish(&mut self, path: &str) -> Result<PublishItemResult, String> {
        publish_one(
            self.session,
            self.gallery,
            self.capabilities,
            false,
            path,
            None,
            &self.state,
            &self.queue_path,
            &mut self.queue,
        )
        .await
    }

    async fn refresh_session(&mut self) -> bool {
        self.refresh_attempts += 1;
        refresh_native_session(self.session).await
    }

    fn record_failure(&mut self, path: &str, error: &str) {
        mark_queue_error(&mut self.queue, path, self.session, &self.gallery.id, error);
    }

    fn invalidate_session(&mut self, error: &str) -> String {
        // Keyring is left alone: this harness never stored the session.
        auth_failure_message(
            error,
            complete_picportal_invalidation(&self.state, Ok(()), None),
        )
    }
}

fn outcome_summary(outcome: &PublicationOutcome) -> serde_json::Value {
    json!({
        "completed": outcome.result.completed,
        "failed": outcome.result.failed,
        "pending": outcome.result.pending,
        "cancelled": outcome.result.cancelled,
        "sessionInvalidated": outcome.session_invalidated,
        "items": outcome.result.items.iter().map(|item| json!({
            "file": Path::new(&item.path).file_name().map(|name| name.to_string_lossy().into_owned()),
            "state": item.state,
            "photoId": item.photo_id,
            "errorCode": item.error.as_ref().map(|error| error.code),
            "errorStatus": item.error.as_ref().and_then(|error| error.status),
            "error": item.error.as_ref().map(|error| error.message.chars().take(300).collect::<String>()),
        })).collect::<Vec<_>>(),
    })
}

async fn photo_count(session: &PicPortalSession, gallery_id: &str) -> u64 {
    session
        .galleries()
        .await
        .expect("gallery list")
        .into_iter()
        .find(|gallery| gallery.id == gallery_id)
        .expect("test gallery listed")
        .photo_count
}

#[tokio::test]
#[ignore = "talks to api.getpicportal.com; needs PICPORTAL_LIVE_EMAIL and PICPORTAL_LIVE_PASSWORD"]
async fn live_publication_round_trip() {
    let (Ok(email), Ok(password)) = (
        std::env::var("PICPORTAL_LIVE_EMAIL"),
        std::env::var("PICPORTAL_LIVE_PASSWORD"),
    ) else {
        eprintln!("PICPORTAL_LIVE_EMAIL/PICPORTAL_LIVE_PASSWORD not set; skipped");
        return;
    };
    let work = tempfile::tempdir().expect("work directory");

    // Connection: wrong password, then the real one.
    let rejected = open_login_session(&email, "not-the-test-password")
        .await
        .err()
        .expect("wrong password is rejected");
    live(
        "login.wrong_password",
        json!({ "code": rejected.code, "status": rejected.status }),
    );
    assert_eq!(rejected.code, "bad_credentials");
    let (session, galleries) = open_login_session(&email, &password).await.expect("login");
    live(
        "login.ok",
        json!({
            "accountId": !session.account_id.is_empty(),
            "galleryCount": galleries.len(),
            "cookieNames": cookie_names(&session),
        }),
    );

    // Restore after restart: the keyring payload is rebuilt into a new client.
    let stored = serde_json::to_string(&stored_session_data(&session).expect("stored session"))
        .expect("serialize stored session");
    let restore = |stored: &str| {
        let stored: StoredSession = serde_json::from_str(stored).expect("stored session");
        let (client, cookies) = production_client(&stored.cookies).expect("client");
        PicPortalSession {
            client,
            base_url: stored.base_url,
            account_id: stored.account_id,
            admin: stored.admin,
            cookies,
            persistent: true,
        }
    };
    let restored = restore(&stored);
    let (account_id, _) = admin_identity(
        session_identity(&restored)
            .await
            .expect("restored identity"),
        &restored.admin,
    )
    .expect("account id");
    assert_eq!(account_id, session.account_id);
    live(
        "restore.ok",
        json!({ "sameAccount": true, "galleryCount": restored.galleries().await.expect("galleries").len() }),
    );

    // Draft gallery without face filtering.
    let title = format!("test-editor-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"));
    let input: CreateGalleryInput = serde_json::from_value(json!({
        "title": title,
        "galleryType": "event",
        "accessMode": "link",
        "status": "draft",
        "faceFilterEnabled": false,
    }))
    .expect("gallery input");
    let created = create_gallery(&restored, input)
        .await
        .expect("create gallery");
    live(
        "gallery.created",
        json!({ "id": created.id, "slug": created.slug, "title": created.title }),
    );
    let relaunched = restore(&stored);
    let matching: Vec<GallerySummary> = relaunched
        .galleries()
        .await
        .expect("galleries after relaunch")
        .into_iter()
        .filter(|gallery| gallery.title == title)
        .collect();
    live(
        "gallery.after_relaunch",
        json!({ "matching": matching.len() }),
    );
    // Share links carry access tokens: only the draft's absence is printed.
    live(
        "gallery.url",
        json!({ "draftGalleryUrl": matching.first().and_then(|gallery| gallery.gallery_url.clone()) }),
    );
    assert!(matching.iter().all(|gallery| gallery.gallery_url.is_none()));
    assert_eq!(matching.len(), 1);
    let gallery = matching.into_iter().next().expect("test gallery");
    assert!(!gallery.face_filter_enabled);

    let capabilities = relaunched
        .processing_capabilities()
        .await
        .expect("capabilities");
    let publish_faces = face_publication_enabled(&gallery, &capabilities, false).expect("faces");
    live(
        "capabilities",
        json!({
            "derivativesTauri": capabilities.derivatives.tauri_enabled,
            "facesTauri": capabilities.faces.tauri_enabled,
            "publishFaces": publish_faces,
        }),
    );
    assert!(!publish_faces);

    let staging = work.path().join("staging");
    fs::create_dir_all(&staging).expect("staging");
    let paths: Vec<String> = (1..=10)
        .map(|index| write_test_jpeg(&staging, index))
        .collect();
    let queue_path = work.path().join("picportal-queue.json");

    // Cancel before the fifth image, then retry the whole selection.
    let mut first = LiveHarness::new(&relaunched, &gallery, &capabilities, queue_path.clone());
    first.cancel_before = Some(4);
    let cancelled = first.run(paths.clone()).await;
    live("publish.cancelled_run", outcome_summary(&cancelled));
    live("publish.cancelled_progress", json!(first.progress));
    assert!(cancelled.result.cancelled);
    assert_eq!(cancelled.result.completed, 4);
    let count_after_cancel = photo_count(&relaunched, &gallery.id).await;
    live("publish.count_after_cancel", json!(count_after_cancel));

    let mut retry = LiveHarness::new(&relaunched, &gallery, &capabilities, queue_path.clone());
    let retried = retry.run(paths.clone()).await;
    live("publish.retry_run", outcome_summary(&retried));
    assert_eq!(retried.result.completed, 10);
    for (before, after) in cancelled
        .result
        .items
        .iter()
        .zip(&retried.result.items)
        .take(4)
    {
        assert_eq!(
            before.photo_id, after.photo_id,
            "retry reused the first upload"
        );
    }
    let count_after_retry = photo_count(&relaunched, &gallery.id).await;
    live("publish.count_after_retry", json!(count_after_retry));
    assert_eq!(count_after_retry, 10);

    // Simulated authentication failure: same account, rejected cookie.
    let (client, cookies) =
        production_client(&["pp_admin=invalid; Path=/; HttpOnly; Secure".to_owned()])
            .expect("client");
    let rejected_session = PicPortalSession {
        client,
        cookies,
        ..relaunched.clone()
    };
    let extra: Vec<String> = (11..=12)
        .map(|index| write_test_jpeg(&staging, index))
        .collect();
    let mut auth = LiveHarness::new(
        &rejected_session,
        &gallery,
        &capabilities,
        work.path().join("auth-queue.json"),
    );
    let auth_outcome = auth.run(extra.clone()).await;
    live("publish.auth_rejected", outcome_summary(&auth_outcome));
    live(
        "publish.auth_refresh_attempts",
        json!(auth.refresh_attempts),
    );
    assert!(auth_outcome.session_invalidated);
    assert_eq!(auth_outcome.result.pending, 1);

    // Simulated network cut: nothing listens on the discard port.
    let offline_session = PicPortalSession {
        base_url: "http://127.0.0.1:9".to_owned(),
        ..relaunched.clone()
    };
    let mut offline = LiveHarness::new(
        &offline_session,
        &gallery,
        &capabilities,
        work.path().join("offline-queue.json"),
    );
    let offline_outcome = offline.run(extra).await;
    live("publish.offline", outcome_summary(&offline_outcome));
    assert!(!offline_outcome.session_invalidated);
    assert_eq!(
        photo_count(&relaunched, &gallery.id).await,
        10,
        "failed runs added nothing"
    );
    let offline_logout = PicPortalError::from(
        complete_picportal_logout(
            &PicPortalState::default(),
            Ok(()),
            remote_logout(&offline_session).await,
        )
        .expect_err("unconfirmed logout"),
    );
    live(
        "logout.offline",
        json!({ "code": offline_logout.code, "retryable": offline_logout.retryable }),
    );

    // Real logout revokes the shared server session.
    let logout_error = remote_logout(&relaunched).await;
    live("logout.ok", json!({ "remoteError": logout_error }));
    assert!(logout_error.is_none());
    let after_logout = session_identity(&restore(&stored)).await.err();
    live(
        "logout.revoked",
        json!({ "status": after_logout.as_deref().and_then(http_status) }),
    );
    assert_eq!(after_logout.as_deref().and_then(http_status), Some(401));
    live(
        "gallery.left_for_cleanup",
        json!({ "id": gallery.id, "slug": gallery.slug }),
    );
}
