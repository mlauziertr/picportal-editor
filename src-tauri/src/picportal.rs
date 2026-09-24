//! PicPortal publication adapter.
//!
//! This module intentionally sits beside RapidRAW rather than changing the
//! platform API.  The editor sends only an explicitly selected exported image,
//! creates delivery derivatives locally, and records destination identity in a
//! durable queue before network work starts.

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::{AppState, face_processing, local_derivatives};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use reqwest::{Client, StatusCode, Url, cookie::Jar, header::SET_COOKIE};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
use keyring::Entry;

pub const PRODUCTION_API_BASE_URL: &str = "https://api.getpicportal.com";
const APP_ORIGIN: &str = "https://getpicportal.com";
const CHUNK_SIZE: usize = 8 * 1024 * 1024;
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
const DERIVATIVE_PIPELINE_VERSION: &str = "picportal-editor-local-derivatives-v1";

#[derive(Default)]
pub struct PicPortalState {
    pub session: Mutex<Option<PicPortalSession>>,
    pub face_runtime: Mutex<Option<face_processing::FaceRuntime>>,
    publish_active: AtomicBool,
    publish_cancelled: AtomicBool,
}

#[derive(Clone)]
pub struct PicPortalSession {
    client: Client,
    base_url: String,
    account_id: String,
    admin: AdminIdentity,
    cookies: Arc<Mutex<Vec<String>>>,
    persistent: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdminIdentity {
    pub email: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GallerySummary {
    pub id: String,
    pub title: String,
    pub slug: String,
    #[serde(default, alias = "photo_count")]
    pub photo_count: u64,
    #[serde(default, alias = "face_filter_enabled")]
    pub face_filter_enabled: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PicPortalLoginResult {
    pub admin: AdminIdentity,
    pub galleries: Vec<GallerySummary>,
    pub persistent: bool,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PicPortalSessionStatus {
    pub connected: bool,
    pub admin: Option<AdminIdentity>,
    pub galleries: Vec<GallerySummary>,
    pub persistent: bool,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PublishResult {
    pub completed: usize,
    pub failed: usize,
    pub cancelled: bool,
    pub items: Vec<PublishItemResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredSession {
    base_url: String,
    account_id: String,
    admin: AdminIdentity,
    cookies: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PublishItemResult {
    pub path: String,
    pub state: String,
    pub photo_id: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct QueueItem {
    path: String,
    api_base_url: String,
    account_id: String,
    gallery_id: String,
    source_sha256: String,
    state: String,
    #[serde(default)]
    photo_id: Option<String>,
    #[serde(default)]
    face_analysis_uploaded: bool,
    #[serde(default)]
    upload_id: Option<String>,
    #[serde(default)]
    upload_offset: u64,
    #[serde(default)]
    last_error: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct QueueFile {
    items: Vec<QueueItem>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProcessingCapabilities {
    derivatives: DerivativeCapabilities,
    faces: FaceCapabilities,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DerivativeCapabilities {
    #[serde(default)]
    tauri_enabled: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FaceCapabilities {
    #[serde(default)]
    tauri_enabled: bool,
    model_id: String,
    model_digest: Option<String>,
    embedding_dim: usize,
    detector_input: usize,
    pipeline_version: String,
    max_faces: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DerivativeDimensions {
    sha256: String,
    width: u32,
    height: u32,
    bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DerivativeManifest {
    version: u8,
    idempotency_key: String,
    producer: DerivativeProducer,
    source: DerivativeDimensions,
    preview: DerivativeDimensions,
    thumbnail: DerivativeDimensions,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DerivativeProducer {
    kind: &'static str,
    app_version: &'static str,
    pipeline_version: &'static str,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FaceAnalysisManifest {
    version: u8,
    idempotency_key: String,
    producer: FaceProducer,
    source: FaceSource,
    model: FaceModel,
    coordinate_system: &'static str,
    faces: Vec<FaceManifestEntry>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FaceProducer {
    kind: &'static str,
    app_version: &'static str,
    pipeline_version: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FaceSource {
    sha256: String,
    bytes: u64,
    width: u32,
    height: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FaceModel {
    id: String,
    digest: String,
    embedding_dim: usize,
    detector_input: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FaceManifestEntry {
    index: usize,
    bbox: FaceBoundingBox,
    confidence: f32,
    embedding: FaceEmbedding,
    thumbnail: FaceThumbnail,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FaceBoundingBox {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FaceEmbedding {
    encoding: &'static str,
    dimension: usize,
    data: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FaceThumbnail {
    field: String,
    sha256: String,
    bytes: u64,
    width: u32,
    height: u32,
    mime: &'static str,
}

#[derive(Debug, Deserialize)]
struct LoginResponse {
    admin: LoginAdmin,
}

#[derive(Debug, Deserialize)]
struct LoginAdmin {
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MeResponse {
    admin: MeAdmin,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MeAdmin {
    #[serde(default)]
    admin_user_id: Option<String>,
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GalleryListResponse {
    galleries: Vec<GallerySummary>,
}

#[derive(Debug, Deserialize)]
struct CreatedGalleryResponse {
    gallery: CreatedGalleryIdentity,
}

#[derive(Debug, Deserialize)]
struct CreatedGalleryIdentity {
    id: String,
    slug: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CreateGalleryInput {
    title: String,
    gallery_type: String,
    access_mode: String,
    status: String,
    face_filter_enabled: bool,
    client_name: Option<String>,
    client_email: Option<String>,
    password: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UploadSession {
    upload_id: String,
    offset: u64,
    #[serde(default)]
    total_bytes: u64,
    #[serde(default)]
    status: String,
    #[serde(default)]
    photo_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PhotoResponse {
    photo: UploadedPhoto,
}

#[derive(Debug, Deserialize)]
struct UploadedPhoto {
    id: String,
}

#[derive(Debug, Deserialize)]
struct DerivativeResponse {
    accepted: bool,
}

#[derive(Debug, Deserialize)]
struct FaceAnalysisResponse {
    status: String,
}

fn endpoint(base_url: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

const KEYRING_SERVICE: &str = "com.getpicportal.PicPortalEditor";
const KEYRING_USER: &str = "picportal-session-v1";
type CookieStore = Arc<Mutex<Vec<String>>>;
const NATIVE_SESSION_REFRESH_PATH: &str = "auth/native/session/refresh";
const SESSION_EXPIRED_MESSAGE: &str = "PicPortal session expired or was revoked; connect again";

fn production_client(stored_cookies: &[String]) -> Result<(Client, CookieStore), String> {
    let cookies = Arc::new(Mutex::new(stored_cookies.to_vec()));
    let jar = Arc::new(Jar::default());
    let url = Url::parse(PRODUCTION_API_BASE_URL)
        .map_err(|error| format!("cannot initialize PicPortal session URL: {error}"))?;
    for cookie in stored_cookies {
        jar.add_cookie_str(cookie, &url);
    }

    Client::builder()
        .cookie_provider(jar)
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(5 * 60))
        .build()
        .map(|client| (client, cookies))
        .map_err(|error| format!("cannot initialize PicPortal network client: {error}"))
}

fn capture_response_cookies(response: &reqwest::Response, cookies: &CookieStore) {
    let Ok(mut current) = cookies.lock() else {
        return;
    };
    for header in response.headers().get_all(SET_COOKIE).iter() {
        let Ok(value) = header.to_str() else {
            continue;
        };
        let Some((name, _)) = value.split_once('=') else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        current.retain(|cookie| {
            cookie
                .split_once('=')
                .is_none_or(|(existing_name, _)| existing_name.trim() != name)
        });
        current.push(value.to_owned());
    }
}

fn secure_storage_error(error: impl std::fmt::Display) -> String {
    format!(
        "Secure PicPortal session storage is unavailable: {error}. Enable the operating system credential store and try again."
    )
}

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
fn load_stored_session() -> Result<Option<StoredSession>, String> {
    let entry = Entry::new(KEYRING_SERVICE, KEYRING_USER).map_err(secure_storage_error)?;
    match entry.get_password() {
        Ok(secret) => serde_json::from_str(&secret).map(Some).map_err(|error| {
            format!("Stored PicPortal session is invalid; connect again ({error})")
        }),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(error) => Err(secure_storage_error(error)),
    }
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn load_stored_session() -> Result<Option<StoredSession>, String> {
    Err("Secure PicPortal session storage is not supported on this platform. Connect for the current session; use a desktop build with the operating system credential store for restart persistence.".to_owned())
}

fn stored_session_data(session: &PicPortalSession) -> Result<StoredSession, String> {
    let cookies = session
        .cookies
        .lock()
        .map_err(|_| "PicPortal cookie state is poisoned".to_owned())?
        .clone();
    Ok(StoredSession {
        base_url: session.base_url.clone(),
        account_id: session.account_id.clone(),
        admin: session.admin.clone(),
        cookies,
    })
}

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
fn store_stored_session(session: &PicPortalSession) -> Result<(), String> {
    let stored = stored_session_data(session)?;
    let secret = serde_json::to_string(&stored)
        .map_err(|error| format!("cannot serialize secure PicPortal session: {error}"))?;
    let entry = Entry::new(KEYRING_SERVICE, KEYRING_USER).map_err(secure_storage_error)?;
    entry.set_password(&secret).map_err(secure_storage_error)
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn store_stored_session(_session: &PicPortalSession) -> Result<(), String> {
    Err("Secure PicPortal session storage is not supported on this platform".to_owned())
}

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
fn clear_stored_session() -> Result<(), String> {
    let entry = Entry::new(KEYRING_SERVICE, KEYRING_USER).map_err(secure_storage_error)?;
    match entry.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(error) => Err(secure_storage_error(error)),
    }
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn clear_stored_session() -> Result<(), String> {
    Ok(())
}

async fn response_error(response: reqwest::Response) -> String {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let detail = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .or_else(|| value.get("message"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| body.chars().take(500).collect());
    format!(
        "HTTP {}: {}",
        status.as_u16(),
        if detail.is_empty() {
            status.to_string()
        } else {
            detail
        }
    )
}

async fn require_success(response: reqwest::Response) -> Result<reqwest::Response, String> {
    if response.status().is_success() {
        Ok(response)
    } else {
        Err(response_error(response).await)
    }
}

fn is_auth_failure(error: &str) -> bool {
    error.contains("HTTP 401:") || error.contains("HTTP 403:")
}

fn clear_session_state(state: &PicPortalState, app: Option<&AppHandle>) {
    if let Ok(mut current) = state.session.lock() {
        current.take();
    }
    if let Some(app) = app {
        let _ = app.emit("picportal-session-invalidated", ());
    }
}

fn invalidate_session(state: &PicPortalState, app: Option<&AppHandle>) {
    let _ = clear_stored_session();
    clear_session_state(state, app);
}

fn complete_picportal_logout(
    state: &PicPortalState,
    secure_storage_clear: Result<(), String>,
    remote_error: Option<String>,
) -> Result<(), String> {
    secure_storage_clear.map_err(|error| {
        format!("PicPortal logout not completed because secure session storage could not be cleared: {error}")
    })?;
    clear_session_state(state, None);
    remote_error.map_or(Ok(()), |error| {
        Err(format!(
            "PicPortal session was removed locally, but remote logout could not be confirmed: {error}"
        ))
    })
}

fn session_error(error: String, state: &PicPortalState, app: Option<&AppHandle>) -> String {
    if is_auth_failure(&error) {
        invalidate_session(state, app);
        SESSION_EXPIRED_MESSAGE.to_owned()
    } else {
        error
    }
}

async fn session_identity(session: &PicPortalSession) -> Result<MeResponse, String> {
    let response = session
        .client
        .get(endpoint(&session.base_url, "admin/me"))
        .header("origin", APP_ORIGIN)
        .send()
        .await
        .map_err(|error| format!("PicPortal identity request failed: {error}"))?;
    capture_response_cookies(&response, &session.cookies);
    require_success(response)
        .await?
        .json::<MeResponse>()
        .await
        .map_err(|error| format!("PicPortal identity response is invalid: {error}"))
}

async fn refresh_native_session(session: &PicPortalSession) -> bool {
    let Ok(response) = session
        .client
        .post(endpoint(&session.base_url, NATIVE_SESSION_REFRESH_PATH))
        .header("origin", APP_ORIGIN)
        .send()
        .await
    else {
        return false;
    };
    capture_response_cookies(&response, &session.cookies);
    response.status().is_success()
}

fn persist_login_session(
    current_session: &mut Option<PicPortalSession>,
    mut next_session: PicPortalSession,
    persist: impl FnOnce(&PicPortalSession) -> Result<(), String>,
) -> Result<PicPortalSession, String> {
    persist(&next_session).map_err(|error| {
        format!(
            "PicPortal login was not completed because secure session storage could not be updated: {error}"
        )
    })?;
    next_session.persistent = true;
    *current_session = Some(next_session.clone());
    Ok(next_session)
}

fn admin_identity(
    me: MeResponse,
    fallback: &AdminIdentity,
) -> Result<(String, AdminIdentity), String> {
    let account_id = me
        .admin
        .admin_user_id
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "PicPortal identity did not include an account id".to_owned())?;
    let email = me.admin.email.unwrap_or_else(|| fallback.email.clone());
    let name = me.admin.name.unwrap_or_else(|| fallback.name.clone());
    Ok((account_id, AdminIdentity { email, name }))
}

fn disconnected_status(message: Option<String>) -> PicPortalSessionStatus {
    PicPortalSessionStatus {
        connected: false,
        admin: None,
        galleries: Vec::new(),
        persistent: false,
        message,
    }
}

#[tauri::command]
pub async fn picportal_login(
    email: String,
    password: String,
    state: State<'_, PicPortalState>,
) -> Result<PicPortalLoginResult, String> {
    if email.trim().is_empty() || password.is_empty() {
        return Err("PicPortal email and password are required".to_owned());
    }
    let (client, cookies) = production_client(&[])?;
    let login_response = client
        .post(endpoint(PRODUCTION_API_BASE_URL, "auth/login"))
        .header("origin", APP_ORIGIN)
        .json(&serde_json::json!({ "email": email.trim(), "password": password }))
        .send()
        .await
        .map_err(|error| format!("PicPortal login failed: {error}"))?;
    capture_response_cookies(&login_response, &cookies);
    let login_response = require_success(login_response).await?;
    let login = login_response
        .json::<LoginResponse>()
        .await
        .map_err(|error| format!("PicPortal login response is invalid: {error}"))?;

    let provisional = PicPortalSession {
        client,
        base_url: PRODUCTION_API_BASE_URL.to_owned(),
        account_id: String::new(),
        admin: AdminIdentity {
            email: email.trim().to_owned(),
            name: email.trim().to_owned(),
        },
        cookies,
        persistent: false,
    };
    let me = session_identity(&provisional).await?;
    let (account_id, admin) = admin_identity(
        me,
        &AdminIdentity {
            email: login.admin.email.unwrap_or_else(|| email.trim().to_owned()),
            name: login.admin.name.unwrap_or_else(|| email.trim().to_owned()),
        },
    )?;
    let session = PicPortalSession {
        account_id,
        admin,
        ..provisional
    };
    let galleries = session.galleries().await?;
    let mut current_session = state
        .session
        .lock()
        .map_err(|_| "PicPortal session lock is poisoned".to_owned())?;
    let session = persist_login_session(&mut current_session, session, store_stored_session)?;
    Ok(PicPortalLoginResult {
        admin: session.admin,
        galleries,
        persistent: true,
        message: None,
    })
}

#[tauri::command]
pub async fn picportal_restore_session(
    state: State<'_, PicPortalState>,
) -> Result<PicPortalSessionStatus, String> {
    let existing = state
        .session
        .lock()
        .map_err(|_| "PicPortal session lock is poisoned".to_owned())?
        .clone();
    if let Some(session) = existing {
        let galleries = match session.galleries().await {
            Ok(galleries) => galleries,
            Err(error) if is_auth_failure(&error) => {
                invalidate_session(&state, None);
                return Ok(disconnected_status(Some(
                    SESSION_EXPIRED_MESSAGE.to_owned(),
                )));
            }
            Err(error) => return Err(error),
        };
        let _ = store_stored_session(&session);
        return Ok(PicPortalSessionStatus {
            connected: true,
            admin: Some(session.admin),
            galleries,
            persistent: session.persistent,
            message: None,
        });
    }
    let Some(stored) = load_stored_session()? else {
        return Ok(disconnected_status(None));
    };
    let (client, cookies) = production_client(&stored.cookies)?;
    let mut session = PicPortalSession {
        client,
        base_url: stored.base_url.clone(),
        account_id: stored.account_id.clone(),
        admin: stored.admin,
        cookies,
        persistent: true,
    };

    let me = match session_identity(&session).await {
        Ok(me) => me,
        Err(error) if is_auth_failure(&error) && refresh_native_session(&session).await => {
            session_identity(&session)
                .await
                .map_err(|retry_error| session_error(retry_error, &state, None))?
        }
        Err(error) if is_auth_failure(&error) => {
            invalidate_session(&state, None);
            return Ok(disconnected_status(Some(
                "PicPortal session expired or was revoked; connect again".to_owned(),
            )));
        }
        Err(error) => return Err(error),
    };
    let (account_id, admin) = admin_identity(me, &session.admin)?;
    if account_id != session.account_id {
        invalidate_session(&state, None);
        return Ok(disconnected_status(Some(
            "PicPortal account changed; connect again".to_owned(),
        )));
    }
    session.admin = admin;
    let galleries = match session.galleries().await {
        Ok(galleries) => galleries,
        Err(error) => {
            let message = session_error(error, &state, None);
            if message == SESSION_EXPIRED_MESSAGE {
                return Ok(disconnected_status(Some(message)));
            }
            return Err(message);
        }
    };
    let message = store_stored_session(&session).err();
    session.persistent = message.is_none();
    *state
        .session
        .lock()
        .map_err(|_| "PicPortal session lock is poisoned".to_owned())? = Some(session.clone());
    Ok(PicPortalSessionStatus {
        connected: true,
        admin: Some(session.admin),
        galleries,
        persistent: session.persistent,
        message,
    })
}

#[tauri::command]
pub async fn picportal_logout(state: State<'_, PicPortalState>) -> Result<(), String> {
    let session = state
        .session
        .lock()
        .map_err(|_| "PicPortal session lock is poisoned".to_owned())?
        .clone();
    let mut remote_error = None;
    if let Some(session) = session {
        match session
            .client
            .post(endpoint(&session.base_url, "auth/logout"))
            .header("origin", APP_ORIGIN)
            .send()
            .await
        {
            Ok(response) => {
                capture_response_cookies(&response, &session.cookies);
                if let Err(error) = require_success(response).await
                    && !is_auth_failure(&error)
                {
                    remote_error = Some(error);
                }
            }
            Err(error) => remote_error = Some(format!("PicPortal logout failed: {error}")),
        }
    }
    complete_picportal_logout(&state, clear_stored_session(), remote_error)
}

#[tauri::command]
pub async fn picportal_galleries(
    state: State<'_, PicPortalState>,
    app: AppHandle,
) -> Result<Vec<GallerySummary>, String> {
    let session = current_session(&state)?;
    let galleries = session
        .galleries()
        .await
        .map_err(|error| session_error(error, &state, Some(&app)))?;
    let _ = store_stored_session(&session);
    Ok(galleries)
}

#[tauri::command]
pub async fn picportal_create_gallery(
    input: CreateGalleryInput,
    state: State<'_, PicPortalState>,
    app: AppHandle,
) -> Result<GallerySummary, String> {
    let session = current_session(&state)?;
    let (title, face_filter_enabled, payload) = create_gallery_payload(input)?;
    let response = session
        .client
        .post(endpoint(&session.base_url, "galleries"))
        .header("origin", APP_ORIGIN)
        .json(&payload)
        .send()
        .await
        .map_err(|error| format!("gallery creation failed: {error}"))
        .map_err(|error| session_error(error, &state, Some(&app)))?;
    capture_response_cookies(&response, &session.cookies);
    let created = require_success(response)
        .await
        .map_err(|error| session_error(error, &state, Some(&app)))?
        .json::<CreatedGalleryResponse>()
        .await
        .map_err(|error| format!("gallery creation response is invalid: {error}"))?
        .gallery;
    let _ = store_stored_session(&session);
    Ok(GallerySummary {
        id: created.id,
        title,
        slug: created.slug,
        photo_count: 0,
        face_filter_enabled,
    })
}

fn create_gallery_payload(
    input: CreateGalleryInput,
) -> Result<(String, bool, serde_json::Value), String> {
    let title = input.title.trim().to_owned();
    if title.is_empty() {
        return Err("gallery title is required".to_owned());
    }
    if !matches!(input.gallery_type.as_str(), "event" | "client") {
        return Err("gallery type must be event or client".to_owned());
    }
    if !matches!(input.access_mode.as_str(), "link" | "password") {
        return Err("gallery access mode must be link or password".to_owned());
    }
    if !matches!(input.status.as_str(), "active" | "draft") {
        return Err("gallery status must be active or draft".to_owned());
    }

    let mut client_name = String::new();
    let mut client_email = String::new();
    if input.gallery_type == "client" {
        client_name = input.client_name.unwrap_or_default().trim().to_owned();
        client_email = input.client_email.unwrap_or_default().trim().to_owned();
        if client_name.is_empty() || client_email.is_empty() {
            return Err("client galleries require a client name and email".to_owned());
        }
    }

    let mut password = String::new();
    if input.access_mode == "password" {
        password = input.password.unwrap_or_default();
        if password.trim().is_empty() {
            return Err("password-protected galleries require a password".to_owned());
        }
    }

    let face_filter_enabled = input.face_filter_enabled;
    let payload = serde_json::json!({
        "title": title,
        "clientName": client_name,
        "clientEmail": client_email,
        "eventDate": "",
        "location": "",
        "galleryType": input.gallery_type,
        "accessMode": input.access_mode,
        "password": password,
        "status": input.status,
        "selectionLimit": serde_json::Value::Null,
        "faceFilterEnabled": face_filter_enabled,
    });
    Ok((title, face_filter_enabled, payload))
}

struct PicPortalActivityGuard<'a> {
    state: &'a PicPortalState,
}

impl Drop for PicPortalActivityGuard<'_> {
    fn drop(&mut self) {
        self.state.publish_active.store(false, Ordering::SeqCst);
        self.state.publish_cancelled.store(false, Ordering::SeqCst);
    }
}

fn begin_picportal_activity(state: &PicPortalState) -> Result<PicPortalActivityGuard<'_>, String> {
    if state.publish_active.swap(true, Ordering::SeqCst) {
        return Err("another PicPortal publication is already running".to_owned());
    }
    state.publish_cancelled.store(false, Ordering::SeqCst);
    Ok(PicPortalActivityGuard { state })
}

pub fn request_picportal_cancellation(app: &AppHandle) -> bool {
    let state = app.state::<PicPortalState>();
    if !state.publish_active.load(Ordering::SeqCst) {
        return false;
    }
    state.publish_cancelled.store(true, Ordering::SeqCst);
    true
}

fn emit_publish_progress(app: &AppHandle, current: usize, total: usize, stage: &str) {
    let _ = app.emit(
        "picportal-progress",
        serde_json::json!({ "current": current, "total": total, "stage": stage }),
    );
}

async fn publish_paths(
    paths: Vec<String>,
    session: &PicPortalSession,
    gallery: &GallerySummary,
    capabilities: &ProcessingCapabilities,
    publish_faces: bool,
    app: &AppHandle,
    state: &PicPortalState,
) -> Result<PublishResult, String> {
    let queue_path = queue_path(app)?;
    let mut queue = load_queue(&queue_path)?;
    let total = paths.len();
    let mut result = PublishResult {
        completed: 0,
        failed: 0,
        cancelled: false,
        items: Vec::new(),
    };
    let mut session_invalidated = false;

    for (index, path) in paths.into_iter().enumerate() {
        if state.publish_cancelled.load(Ordering::SeqCst) {
            result.cancelled = true;
            break;
        }
        emit_publish_progress(app, index, total, "uploading");
        let item_result = publish_one(
            session,
            gallery,
            capabilities,
            publish_faces,
            &path,
            app,
            state,
            &queue_path,
            &mut queue,
        )
        .await;
        match item_result {
            Ok(item) => {
                result.completed += 1;
                result.items.push(item);
            }
            Err(error) => {
                let authentication_failed = is_auth_failure(&error);
                result.failed += 1;
                mark_queue_error(&mut queue, &path, session, &gallery.id, &error);
                result.items.push(PublishItemResult {
                    path,
                    state: if state.publish_cancelled.load(Ordering::SeqCst) {
                        "cancelled".to_owned()
                    } else {
                        "failed".to_owned()
                    },
                    photo_id: None,
                    error: Some(if authentication_failed {
                        SESSION_EXPIRED_MESSAGE.to_owned()
                    } else {
                        error
                    }),
                });
                if authentication_failed {
                    invalidate_session(state, Some(app));
                    session_invalidated = true;
                }
                if state.publish_cancelled.load(Ordering::SeqCst) {
                    result.cancelled = true;
                    break;
                }
                if session_invalidated {
                    break;
                }
            }
        }
        emit_publish_progress(app, index + 1, total, "uploading");
        if session_invalidated {
            break;
        }
    }
    if state.publish_cancelled.load(Ordering::SeqCst) {
        result.cancelled = true;
    }
    save_queue(&queue_path, &queue)?;
    if !session_invalidated {
        let _ = store_stored_session(session);
    }
    Ok(result)
}

fn normalize_picportal_output_format(output_format: &str) -> Result<String, String> {
    let extension = output_format.trim().to_ascii_lowercase();
    let extension = if extension == "jpeg" {
        "jpg".to_owned()
    } else {
        extension
    };
    if matches!(extension.as_str(), "jpg" | "png" | "webp") {
        Ok(extension)
    } else {
        Err("PicPortal export supports JPEG, PNG or WebP output only".to_owned())
    }
}

#[allow(clippy::too_many_arguments)]
fn direct_export_operation_id(
    session: &PicPortalSession,
    gallery_id: &str,
    paths: &[String],
    base_origin_folders: &[String],
    export_settings: &crate::export_processing::ExportSettings,
    output_format: &str,
    current_edit_path: &Option<String>,
    current_edit_adjustments: &Option<serde_json::Value>,
    include_face_analysis: bool,
) -> Result<uuid::Uuid, String> {
    let operation = serde_json::to_vec(&(
        &session.base_url,
        &session.account_id,
        gallery_id,
        paths,
        base_origin_folders,
        export_settings,
        output_format,
        current_edit_path,
        current_edit_adjustments,
        include_face_analysis,
    ))
    .map_err(|error| format!("cannot identify PicPortal export: {error}"))?;
    Ok(uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, &operation))
}

fn prepare_staging_directory(app: &AppHandle, operation_id: uuid::Uuid) -> Result<PathBuf, String> {
    let root = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("cannot resolve PicPortal staging directory: {error}"))?
        .join("picportal-staging")
        .join(operation_id.to_string());
    if root.exists() {
        fs::remove_dir_all(&root)
            .map_err(|error| format!("cannot reset PicPortal staging directory: {error}"))?;
    }
    fs::create_dir_all(&root)
        .map_err(|error| format!("cannot create PicPortal staging directory: {error}"))?;
    Ok(root)
}

fn staged_exports(root: &Path, extension: &str, expected: usize) -> Result<Vec<String>, String> {
    let mut paths = Vec::new();
    for entry in walkdir::WalkDir::new(root) {
        let entry =
            entry.map_err(|error| format!("cannot inspect PicPortal staging output: {error}"))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let matches_extension = path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case(extension));
        if matches_extension {
            paths.push(path.to_string_lossy().to_string());
        }
    }
    paths.sort();
    if paths.len() != expected {
        return Err(format!(
            "PicPortal rendering produced {} outputs for {} selected images; no upload was started",
            paths.len(),
            expected
        ));
    }
    Ok(paths)
}

#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn picportal_export(
    paths: Vec<String>,
    base_origin_folders: Vec<String>,
    export_settings: crate::export_processing::ExportSettings,
    output_format: String,
    current_edit_path: Option<String>,
    current_edit_adjustments: Option<serde_json::Value>,
    gallery_id: String,
    include_face_analysis: bool,
    app: AppHandle,
    state: State<'_, PicPortalState>,
    export_state: State<'_, AppState>,
) -> Result<PublishResult, String> {
    if paths.is_empty() {
        return Err("select at least one image for PicPortal export".to_owned());
    }
    let output_format = normalize_picportal_output_format(&output_format)?;
    if export_settings.export_masks {
        return Err(
            "PicPortal export does not upload separate mask files; turn off Export masks"
                .to_owned(),
        );
    }
    let session = current_session(&state)?;
    let galleries = session
        .galleries()
        .await
        .map_err(|error| session_error(error, &state, Some(&app)))?;
    let gallery = galleries
        .into_iter()
        .find(|gallery| gallery.id == gallery_id)
        .ok_or_else(|| "selected PicPortal gallery no longer exists".to_owned())?;
    let capabilities = session
        .processing_capabilities()
        .await
        .map_err(|error| session_error(error, &state, Some(&app)))?;
    if !capabilities.derivatives.tauri_enabled {
        return Err("PicPortal local derivative processing is disabled by the server".to_owned());
    }
    let publish_faces = face_publication_enabled(&gallery, &capabilities, include_face_analysis)?;
    let operation_id = direct_export_operation_id(
        &session,
        &gallery.id,
        &paths,
        &base_origin_folders,
        &export_settings,
        &output_format,
        &current_edit_path,
        &current_edit_adjustments,
        include_face_analysis,
    )?;
    let _activity = begin_picportal_activity(&state)?;
    let staging_root = prepare_staging_directory(&app, operation_id)?;

    let mut staging_settings = export_settings;
    staging_settings.destination_type = Some("customFolder".to_owned());
    staging_settings.subfolder = None;
    let (completion_tx, completion_rx) = tokio::sync::oneshot::channel();
    crate::export_processing::export_images_impl(
        paths.clone(),
        staging_root.to_string_lossy().to_string(),
        false,
        base_origin_folders,
        staging_settings,
        output_format.clone(),
        crate::export_processing::ExportAdjustmentsMode::UseSidecars {
            active_path: current_edit_path,
            active_adjustments: current_edit_adjustments,
        },
        export_state,
        app.clone(),
        Some(completion_tx),
        false,
    )
    .await?;
    match completion_rx.await {
        Ok(Ok(())) => {}
        Ok(Err(errors)) => {
            if state.publish_cancelled.load(Ordering::SeqCst) {
                return Err(
                    "PicPortal export cancelled; generated files were kept for a safe retry"
                        .to_owned(),
                );
            }
            return Err(format!(
                "PicPortal rendering completed with {errors} error(s); no upload was started"
            ));
        }
        Err(_) if state.publish_cancelled.load(Ordering::SeqCst) => {
            return Err(
                "PicPortal export cancelled; generated files were kept for a safe retry".to_owned(),
            );
        }
        Err(_) => {
            return Err(
                "PicPortal rendering task ended unexpectedly; no upload was started".to_owned(),
            );
        }
    }
    ensure_publish_not_cancelled(&state)?;
    let staged_paths = staged_exports(&staging_root, &output_format, paths.len())?;
    let result = publish_paths(
        staged_paths,
        &session,
        &gallery,
        &capabilities,
        publish_faces,
        &app,
        &state,
    )
    .await
    .map_err(|error| session_error(error, &state, Some(&app)))?;
    if result.failed == 0 && !result.cancelled {
        let _ = fs::remove_dir_all(&staging_root);
    }
    Ok(result)
}

fn current_session(state: &PicPortalState) -> Result<PicPortalSession, String> {
    state
        .session
        .lock()
        .map_err(|_| "PicPortal session lock is poisoned".to_owned())?
        .clone()
        .ok_or_else(|| "log in to PicPortal before using publication".to_owned())
}

impl PicPortalSession {
    async fn galleries(&self) -> Result<Vec<GallerySummary>, String> {
        let response = self
            .client
            .get(endpoint(&self.base_url, "galleries"))
            .header("origin", APP_ORIGIN)
            .send()
            .await
            .map_err(|error| format!("gallery request failed: {error}"))?;
        capture_response_cookies(&response, &self.cookies);
        let response = require_success(response).await?;
        let value = response
            .json::<serde_json::Value>()
            .await
            .map_err(|error| format!("gallery response is invalid: {error}"))?;
        if let Ok(payload) = serde_json::from_value::<GalleryListResponse>(value.clone()) {
            return Ok(payload.galleries);
        }
        serde_json::from_value(value)
            .map_err(|error| format!("gallery response is invalid: {error}"))
    }

    async fn processing_capabilities(&self) -> Result<ProcessingCapabilities, String> {
        let response = self
            .client
            .get(endpoint(&self.base_url, "admin/processing-capabilities"))
            .header("origin", APP_ORIGIN)
            .send()
            .await
            .map_err(|error| format!("processing capabilities request failed: {error}"))?;
        capture_response_cookies(&response, &self.cookies);
        require_success(response)
            .await?
            .json()
            .await
            .map_err(|error| format!("processing capabilities response is invalid: {error}"))
    }
}

fn ensure_publish_not_cancelled(state: &PicPortalState) -> Result<(), String> {
    if state.publish_cancelled.load(Ordering::SeqCst) {
        Err("PicPortal export cancelled; generated files were kept for a safe retry".to_owned())
    } else {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
async fn publish_one(
    session: &PicPortalSession,
    gallery: &GallerySummary,
    capabilities: &ProcessingCapabilities,
    publish_faces: bool,
    path: &str,
    app: &AppHandle,
    state: &PicPortalState,
    queue_path: &Path,
    queue: &mut QueueFile,
) -> Result<PublishItemResult, String> {
    ensure_publish_not_cancelled(state)?;
    let source_path = PathBuf::from(path);
    validate_export_path(&source_path)?;
    let source = fs::read(&source_path)
        .map_err(|error| format!("cannot read export {}: {error}", source_path.display()))?;
    let source_sha256 = local_derivatives::sha256_hex(&source);
    let existing = queue
        .items
        .iter()
        .find(|item| queue_identity_matches(item, path, session, &gallery.id, &source_sha256))
        .cloned();
    if let Some(existing) = existing.as_ref()
        && existing.state == "uploaded"
    {
        if let Some(photo_id) = completed_face_analysis_photo(existing, publish_faces)? {
            let analysis = analyze_faces(&source, capabilities, app, state)?;
            upload_faces_for_photo(
                session,
                &gallery.id,
                &photo_id,
                &source_sha256,
                source.len() as u64,
                &analysis,
            )
            .await?;
            if let Some(item) = queue.items.iter_mut().find(|item| {
                queue_identity_matches(item, path, session, &gallery.id, &source_sha256)
            }) {
                item.face_analysis_uploaded = true;
                item.last_error = None;
            }
            save_queue(queue_path, queue)?;
        }
        return Ok(PublishItemResult {
            path: path.to_owned(),
            state: existing.state.clone(),
            photo_id: existing.photo_id.clone(),
            error: None,
        });
    }
    ensure_publish_not_cancelled(state)?;
    let derivatives = local_derivatives::generate_local_derivatives(&source)?;
    if existing.is_none() {
        upsert_queue(
            queue,
            QueueItem {
                path: path.to_owned(),
                api_base_url: session.base_url.clone(),
                account_id: session.account_id.clone(),
                gallery_id: gallery.id.clone(),
                source_sha256: derivatives.source_sha256.clone(),
                state: "queued".to_owned(),
                photo_id: None,
                face_analysis_uploaded: false,
                upload_id: None,
                upload_offset: 0,
                last_error: None,
            },
        );
    } else if let Some(item) = queue.items.iter_mut().find(|item| {
        queue_identity_matches(item, path, session, &gallery.id, &derivatives.source_sha256)
    }) {
        item.state = "uploading".to_owned();
        item.last_error = None;
    }
    save_queue(queue_path, queue)?;

    let face_analysis = if publish_faces {
        ensure_publish_not_cancelled(state)?;
        Some(analyze_faces(&source, capabilities, app, state)?)
    } else {
        None
    };

    let filename = source_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("export.jpg");
    ensure_publish_not_cancelled(state)?;
    let photo_id = upload_original(
        session,
        &gallery.id,
        filename,
        derivatives.source_mime,
        &source,
        &derivatives.source_sha256,
        path,
        queue_path,
        queue,
        state,
    )
    .await?;
    let idempotency_key = idempotency_key(
        &session.base_url,
        &session.account_id,
        &gallery.id,
        &photo_id,
        &derivatives.source_sha256,
    );
    let manifest = DerivativeManifest {
        version: 1,
        idempotency_key: idempotency_key.clone(),
        producer: DerivativeProducer {
            kind: "picportal-editor",
            app_version: APP_VERSION,
            pipeline_version: DERIVATIVE_PIPELINE_VERSION,
        },
        source: DerivativeDimensions {
            sha256: derivatives.source_sha256.clone(),
            width: derivatives.source_width,
            height: derivatives.source_height,
            bytes: source.len() as u64,
        },
        preview: derivative_dimensions(&derivatives.preview),
        thumbnail: derivative_dimensions(&derivatives.thumbnail),
    };
    ensure_publish_not_cancelled(state)?;
    upload_derivatives(
        session,
        &gallery.id,
        &photo_id,
        &manifest,
        &derivatives.preview.bytes,
        &derivatives.thumbnail.bytes,
    )
    .await?;
    if let Some(analysis) = face_analysis {
        ensure_publish_not_cancelled(state)?;
        upload_faces_for_photo(
            session,
            &gallery.id,
            &photo_id,
            &derivatives.source_sha256,
            source.len() as u64,
            &analysis,
        )
        .await?;
    }
    if let Some(item) = queue.items.iter_mut().find(|item| {
        queue_identity_matches(item, path, session, &gallery.id, &derivatives.source_sha256)
    }) {
        item.state = "uploaded".to_owned();
        item.photo_id = Some(photo_id.clone());
        item.face_analysis_uploaded = publish_faces;
        item.upload_id = None;
        item.upload_offset = source.len() as u64;
        item.last_error = None;
    }
    save_queue(queue_path, queue)?;
    Ok(PublishItemResult {
        path: path.to_owned(),
        state: "uploaded".to_owned(),
        photo_id: Some(photo_id),
        error: None,
    })
}

fn face_publication_enabled(
    gallery: &GallerySummary,
    capabilities: &ProcessingCapabilities,
    requested: bool,
) -> Result<bool, String> {
    if !requested {
        return Ok(false);
    }
    if !gallery.face_filter_enabled {
        return Err("selected PicPortal gallery does not allow face analysis".to_owned());
    }
    if !capabilities.faces.tauri_enabled {
        return Err(
            "PicPortal local face processing is disabled by the server; no remote fallback was used"
                .to_owned(),
        );
    }
    Ok(true)
}

fn completed_face_analysis_photo(
    item: &QueueItem,
    publish_faces: bool,
) -> Result<Option<String>, String> {
    if item.state != "uploaded" || !publish_faces || item.face_analysis_uploaded {
        return Ok(None);
    }
    item.photo_id
        .as_ref()
        .filter(|photo_id| !photo_id.trim().is_empty())
        .cloned()
        .map(Some)
        .ok_or_else(|| "completed PicPortal publication has no photo id".to_owned())
}

fn analyze_faces(
    source: &[u8],
    capabilities: &ProcessingCapabilities,
    app: &AppHandle,
    state: &PicPortalState,
) -> Result<face_processing::LocalFaceAnalysis, String> {
    let mut runtime = state
        .face_runtime
        .lock()
        .map_err(|_| "face runtime lock is poisoned".to_owned())?;
    let runtime = face_processing::ensure_runtime(&mut runtime, app)?;
    let analysis = runtime.analyze(source)?;
    if analysis.model_id != capabilities.faces.model_id
        || analysis.pipeline_version != capabilities.faces.pipeline_version
        || analysis.embedding_dimension != capabilities.faces.embedding_dim
        || analysis.detector_input != capabilities.faces.detector_input
        || analysis.faces.len() > capabilities.faces.max_faces
        || capabilities
            .faces
            .model_digest
            .as_deref()
            .is_some_and(|digest| digest != analysis.model_digest)
    {
        return Err("PicPortal face model contract is incompatible with the server".to_owned());
    }
    Ok(analysis)
}

fn validate_export_path(path: &Path) -> Result<(), String> {
    let metadata =
        fs::metadata(path).map_err(|_| format!("export does not exist: {}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("export does not exist: {}", path.display()));
    }
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "jpg" | "jpeg" | "png" | "webp") {
        return Err("PicPortal publication accepts an exported JPEG, PNG or WebP; RAW files are never uploaded by this workflow".to_owned());
    }
    if metadata.len() == 0 || metadata.len() > local_derivatives::MAX_SOURCE_BYTES {
        return Err(format!(
            "source must be between 1 and {} bytes",
            local_derivatives::MAX_SOURCE_BYTES
        ));
    }
    Ok(())
}

fn derivative_dimensions(derivative: &local_derivatives::LocalDerivative) -> DerivativeDimensions {
    DerivativeDimensions {
        sha256: derivative.sha256.clone(),
        width: derivative.width,
        height: derivative.height,
        bytes: derivative.bytes.len() as u64,
    }
}

fn idempotency_key(
    api_base_url: &str,
    account_id: &str,
    gallery_id: &str,
    photo_id: &str,
    source_sha256: &str,
) -> String {
    let operation = serde_json::to_vec(&(
        api_base_url,
        account_id,
        gallery_id,
        photo_id,
        source_sha256,
        DERIVATIVE_PIPELINE_VERSION,
    ))
    .expect("idempotency operation identity is serializable");
    uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, &operation).to_string()
}

fn face_manifest(
    analysis: &face_processing::LocalFaceAnalysis,
    idempotency_key: &str,
    source_bytes: u64,
) -> FaceAnalysisManifest {
    FaceAnalysisManifest {
        version: 1,
        idempotency_key: idempotency_key.to_owned(),
        producer: FaceProducer {
            kind: "picportal-editor",
            app_version: APP_VERSION,
            pipeline_version: analysis.pipeline_version.clone(),
        },
        source: FaceSource {
            sha256: analysis.source_sha256.clone(),
            bytes: source_bytes,
            width: analysis.source_width,
            height: analysis.source_height,
        },
        model: FaceModel {
            id: analysis.model_id.clone(),
            digest: analysis.model_digest.clone(),
            embedding_dim: analysis.embedding_dimension,
            detector_input: analysis.detector_input,
        },
        coordinate_system: "normalized-top-left",
        faces: analysis
            .faces
            .iter()
            .enumerate()
            .map(|(index, face)| FaceManifestEntry {
                index,
                bbox: FaceBoundingBox {
                    x: face.bbox.x,
                    y: face.bbox.y,
                    width: face.bbox.width,
                    height: face.bbox.height,
                },
                confidence: face.confidence,
                embedding: FaceEmbedding {
                    encoding: "float32-le-base64",
                    dimension: face.embedding.len(),
                    data: BASE64.encode(
                        face.embedding
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<_>>(),
                    ),
                },
                thumbnail: FaceThumbnail {
                    field: format!("face-{index}"),
                    sha256: face.thumbnail_sha256.clone(),
                    bytes: face.thumbnail.len() as u64,
                    width: 220,
                    height: 220,
                    mime: "image/webp",
                },
            })
            .collect(),
    }
}

fn clear_queue_upload_session(
    queue: &mut QueueFile,
    path: &str,
    session: &PicPortalSession,
    gallery_id: &str,
    source_sha256: &str,
) {
    if let Some(item) = queue
        .items
        .iter_mut()
        .find(|item| queue_identity_matches(item, path, session, gallery_id, source_sha256))
    {
        item.upload_id = None;
        item.upload_offset = 0;
    }
}

fn update_queue_upload_session(
    queue: &mut QueueFile,
    path: &str,
    session: &PicPortalSession,
    gallery_id: &str,
    source_sha256: &str,
    upload_id: &str,
    offset: u64,
) {
    if let Some(item) = queue
        .items
        .iter_mut()
        .find(|item| queue_identity_matches(item, path, session, gallery_id, source_sha256))
    {
        item.state = "uploading".to_owned();
        item.upload_id = Some(upload_id.to_owned());
        item.upload_offset = offset;
        item.last_error = None;
    }
}

fn mark_original_uploaded(
    queue: &mut QueueFile,
    path: &str,
    session: &PicPortalSession,
    gallery_id: &str,
    source_sha256: &str,
    photo_id: &str,
    offset: u64,
) {
    if let Some(item) = queue
        .items
        .iter_mut()
        .find(|item| queue_identity_matches(item, path, session, gallery_id, source_sha256))
    {
        item.state = "original_uploaded".to_owned();
        item.photo_id = Some(photo_id.to_owned());
        item.upload_id = None;
        item.upload_offset = offset;
        item.last_error = None;
    }
}

#[allow(clippy::too_many_arguments)]
async fn upload_original(
    session: &PicPortalSession,
    gallery_id: &str,
    filename: &str,
    content_type: &str,
    source: &[u8],
    source_sha256: &str,
    path: &str,
    queue_path: &Path,
    queue: &mut QueueFile,
    state: &PicPortalState,
) -> Result<String, String> {
    ensure_publish_not_cancelled(state)?;
    let total_bytes = source.len() as u64;
    if let Some(existing) = queue
        .items
        .iter()
        .find(|item| queue_identity_matches(item, path, session, gallery_id, source_sha256))
        && let Some(photo_id) = existing.photo_id.as_deref()
    {
        return Ok(photo_id.to_owned());
    }

    let start_url = endpoint(
        &session.base_url,
        &format!("galleries/{gallery_id}/photos/uploads"),
    );
    let requested_upload_id = queue
        .items
        .iter()
        .find(|item| queue_identity_matches(item, path, session, gallery_id, source_sha256))
        .and_then(|item| item.upload_id.clone());
    let start_request = |upload_id: Option<&str>| {
        let mut payload = serde_json::json!({
            "filename": filename,
            "contentType": content_type,
            "totalBytes": total_bytes,
        });
        if let Some(upload_id) = upload_id {
            payload["uploadId"] = serde_json::Value::String(upload_id.to_owned());
        }
        session
            .client
            .post(&start_url)
            .header("origin", APP_ORIGIN)
            .json(&payload)
    };
    let mut start = start_request(requested_upload_id.as_deref())
        .send()
        .await
        .map_err(|error| format!("upload session failed: {error}"))?;
    capture_response_cookies(&start, &session.cookies);
    if requested_upload_id.is_some()
        && matches!(start.status(), StatusCode::NOT_FOUND | StatusCode::CONFLICT)
    {
        // A server-side session may have expired. Retain the source identity but
        // create a fresh session rather than silently changing destination.
        let _ = start.text().await;
        clear_queue_upload_session(queue, path, session, gallery_id, source_sha256);
        start = start_request(None)
            .send()
            .await
            .map_err(|error| format!("new upload session failed: {error}"))?;
        capture_response_cookies(&start, &session.cookies);
    }
    if start.status() == StatusCode::NOT_FOUND {
        let body = start.text().await.unwrap_or_default();
        if body.contains("Route POST:") && body.contains("/photos/uploads not found") {
            let photo_id =
                upload_multipart(session, gallery_id, filename, content_type, source).await?;
            mark_original_uploaded(
                queue,
                path,
                session,
                gallery_id,
                source_sha256,
                &photo_id,
                total_bytes,
            );
            save_queue(queue_path, queue)?;
            return Ok(photo_id);
        }
        return Err(format!(
            "resumable upload route rejected the request: {body}"
        ));
    }
    let session_payload = require_success(start)
        .await?
        .json::<UploadSession>()
        .await
        .map_err(|error| format!("upload session response is invalid: {error}"))?;
    if session_payload.total_bytes != 0 && session_payload.total_bytes != total_bytes {
        return Err("PicPortal upload session has a different source size".to_owned());
    }
    if session_payload.offset > total_bytes {
        return Err("PicPortal upload session offset exceeds the source".to_owned());
    }
    if session_payload.status == "completed" {
        let photo_id = session_payload
            .photo_id
            .filter(|id| !id.trim().is_empty())
            .ok_or_else(|| "completed PicPortal upload has no photo id".to_owned())?;
        mark_original_uploaded(
            queue,
            path,
            session,
            gallery_id,
            source_sha256,
            &photo_id,
            total_bytes,
        );
        save_queue(queue_path, queue)?;
        return Ok(photo_id);
    }

    let upload_id = session_payload.upload_id.trim().to_owned();
    if upload_id.is_empty() {
        return Err("PicPortal upload session has no upload id".to_owned());
    }
    let mut offset = session_payload.offset;
    update_queue_upload_session(
        queue,
        path,
        session,
        gallery_id,
        source_sha256,
        &upload_id,
        offset,
    );
    save_queue(queue_path, queue)?;
    while offset < total_bytes {
        let start_offset = usize::try_from(offset)
            .map_err(|_| "upload offset does not fit local memory".to_owned())?;
        let end_offset = (start_offset + CHUNK_SIZE).min(source.len());
        let chunk = source[start_offset..end_offset].to_vec();
        let end = end_offset as u64 - 1;
        ensure_publish_not_cancelled(state)?;
        let response = session
            .client
            .patch(endpoint(
                &session.base_url,
                &format!("galleries/{gallery_id}/photos/uploads/{upload_id}"),
            ))
            .header("origin", APP_ORIGIN)
            .header("content-type", "application/octet-stream")
            .header(
                "content-range",
                format!("bytes {offset}-{end}/{total_bytes}"),
            )
            .body(chunk)
            .send()
            .await
            .map_err(|error| format!("upload chunk failed: {error}"))?;
        capture_response_cookies(&response, &session.cookies);
        let payload = require_success(response)
            .await?
            .json::<UploadSession>()
            .await
            .map_err(|error| format!("upload chunk response is invalid: {error}"))?;
        if payload.offset <= offset || payload.offset > total_bytes {
            return Err("PicPortal upload did not advance to a valid durable offset".to_owned());
        }
        offset = payload.offset;
        update_queue_upload_session(
            queue,
            path,
            session,
            gallery_id,
            source_sha256,
            &upload_id,
            offset,
        );
        save_queue(queue_path, queue)?;
    }
    ensure_publish_not_cancelled(state)?;
    let complete = session
        .client
        .post(endpoint(
            &session.base_url,
            &format!("galleries/{gallery_id}/photos/uploads/{upload_id}/complete"),
        ))
        .header("origin", APP_ORIGIN)
        .json(&serde_json::json!({ "sha256": source_sha256, "deferProcessing": true }))
        .send()
        .await
        .map_err(|error| format!("upload completion failed: {error}"))?;
    capture_response_cookies(&complete, &session.cookies);
    let photo = require_success(complete)
        .await?
        .json::<PhotoResponse>()
        .await
        .map_err(|error| format!("upload completion response is invalid: {error}"))?
        .photo;
    if photo.id.trim().is_empty() {
        return Err("PicPortal returned an empty photo id".to_owned());
    }
    mark_original_uploaded(
        queue,
        path,
        session,
        gallery_id,
        source_sha256,
        &photo.id,
        total_bytes,
    );
    save_queue(queue_path, queue)?;
    Ok(photo.id)
}

async fn upload_multipart(
    session: &PicPortalSession,
    gallery_id: &str,
    filename: &str,
    content_type: &str,
    source: &[u8],
) -> Result<String, String> {
    let part = reqwest::multipart::Part::bytes(source.to_vec())
        .file_name(filename.to_owned())
        .mime_str(content_type)
        .map_err(|error| error.to_string())?;
    let form = reqwest::multipart::Form::new().part("file", part);
    let response = session
        .client
        .post(endpoint(
            &session.base_url,
            &format!("galleries/{gallery_id}/photos?deferProcessing=true"),
        ))
        .header("origin", APP_ORIGIN)
        .multipart(form)
        .send()
        .await
        .map_err(|error| format!("multipart upload failed: {error}"))?;
    capture_response_cookies(&response, &session.cookies);
    let photo = require_success(response)
        .await?
        .json::<PhotoResponse>()
        .await
        .map_err(|error| format!("multipart upload response is invalid: {error}"))?
        .photo;
    if photo.id.trim().is_empty() {
        return Err("PicPortal returned an empty photo id".to_owned());
    }
    Ok(photo.id)
}

async fn upload_derivatives(
    session: &PicPortalSession,
    gallery_id: &str,
    photo_id: &str,
    manifest: &DerivativeManifest,
    preview: &[u8],
    thumbnail: &[u8],
) -> Result<(), String> {
    let preview_part = reqwest::multipart::Part::bytes(preview.to_vec())
        .file_name("preview.webp")
        .mime_str("image/webp")
        .map_err(|error| error.to_string())?;
    let thumbnail_part = reqwest::multipart::Part::bytes(thumbnail.to_vec())
        .file_name("thumbnail.webp")
        .mime_str("image/webp")
        .map_err(|error| error.to_string())?;
    let manifest_json = serde_json::to_string(manifest).map_err(|error| error.to_string())?;
    let form = reqwest::multipart::Form::new()
        .part("preview", preview_part)
        .part("thumbnail", thumbnail_part)
        .text("manifest", manifest_json);
    let response = session
        .client
        .post(endpoint(
            &session.base_url,
            &format!("galleries/{gallery_id}/photos/{photo_id}/derivatives"),
        ))
        .header("origin", APP_ORIGIN)
        .multipart(form)
        .send()
        .await
        .map_err(|error| format!("derivative upload failed: {error}"))?;
    capture_response_cookies(&response, &session.cookies);
    let accepted = require_success(response)
        .await?
        .json::<DerivativeResponse>()
        .await
        .map_err(|error| format!("derivative response is invalid: {error}"))?;
    if !accepted.accepted {
        return Err("PicPortal rejected the local derivative manifest".to_owned());
    }
    Ok(())
}

async fn upload_face_analysis(
    session: &PicPortalSession,
    gallery_id: &str,
    photo_id: &str,
    manifest: &FaceAnalysisManifest,
    thumbnails: &[Vec<u8>],
) -> Result<(), String> {
    let manifest_json = serde_json::to_string(manifest).map_err(|error| error.to_string())?;
    let mut form = reqwest::multipart::Form::new().text("manifest", manifest_json);
    for (index, thumbnail) in thumbnails.iter().enumerate() {
        let part = reqwest::multipart::Part::bytes(thumbnail.clone())
            .file_name(format!("face-{index}.webp"))
            .mime_str("image/webp")
            .map_err(|error| error.to_string())?;
        form = form.part(format!("face-{index}"), part);
    }
    let response = session
        .client
        .post(endpoint(
            &session.base_url,
            &format!("galleries/{gallery_id}/photos/{photo_id}/face-analysis"),
        ))
        .header("origin", APP_ORIGIN)
        .multipart(form)
        .send()
        .await
        .map_err(|error| format!("face analysis upload failed: {error}"))?;
    capture_response_cookies(&response, &session.cookies);
    let accepted = require_success(response)
        .await?
        .json::<FaceAnalysisResponse>()
        .await
        .map_err(|error| format!("face analysis response is invalid: {error}"))?;
    if !matches!(accepted.status.as_str(), "accepted" | "ready") {
        return Err("PicPortal rejected the local face analysis".to_owned());
    }
    Ok(())
}

async fn upload_faces_for_photo(
    session: &PicPortalSession,
    gallery_id: &str,
    photo_id: &str,
    source_sha256: &str,
    source_bytes: u64,
    analysis: &face_processing::LocalFaceAnalysis,
) -> Result<(), String> {
    let idempotency_key = idempotency_key(
        &session.base_url,
        &session.account_id,
        gallery_id,
        photo_id,
        source_sha256,
    );
    let manifest = face_manifest(analysis, &idempotency_key, source_bytes);
    let thumbnails = analysis
        .faces
        .iter()
        .map(|face| face.thumbnail.clone())
        .collect::<Vec<_>>();
    upload_face_analysis(session, gallery_id, photo_id, &manifest, &thumbnails).await
}

fn queue_path(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_data_dir()
        .map(|path| path.join("picportal-queue.json"))
        .map_err(|error| format!("cannot resolve PicPortal queue: {error}"))
}

fn load_queue(path: &Path) -> Result<QueueFile, String> {
    let temporary = path.with_extension("json.tmp");
    let backup = path.with_extension("json.bak");
    let mut invalid_queue = None;
    for candidate in [temporary.as_path(), path, backup.as_path()] {
        if !candidate.is_file() {
            continue;
        }
        let bytes =
            fs::read(candidate).map_err(|error| format!("cannot read PicPortal queue: {error}"))?;
        match serde_json::from_slice(&bytes) {
            Ok(queue) => {
                if candidate == temporary.as_path() {
                    commit_temporary_queue(path)?;
                }
                return Ok(queue);
            }
            Err(error) => invalid_queue = Some(error),
        }
    }
    invalid_queue.map_or_else(
        || Ok(QueueFile::default()),
        |error| Err(format!("PicPortal queue is invalid: {error}")),
    )
}

fn save_queue(path: &Path, queue: &QueueFile) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "PicPortal queue has no parent directory".to_owned())?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("cannot create PicPortal queue directory: {error}"))?;
    let temporary = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(queue)
        .map_err(|error| format!("cannot serialize PicPortal queue: {error}"))?;
    let mut temporary_file = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| format!("cannot open PicPortal queue: {error}"))?;
    temporary_file
        .write_all(&bytes)
        .map_err(|error| format!("cannot write PicPortal queue: {error}"))?;
    temporary_file
        .sync_all()
        .map_err(|error| format!("cannot flush PicPortal queue: {error}"))?;
    drop(temporary_file);

    commit_temporary_queue(path)
}

fn commit_temporary_queue(path: &Path) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "PicPortal queue has no parent directory".to_owned())?;
    let temporary = path.with_extension("json.tmp");
    let backup = path.with_extension("json.bak");
    if path.exists() {
        if backup.exists() {
            fs::remove_file(&backup)
                .map_err(|error| format!("cannot clear PicPortal queue backup: {error}"))?;
        }
        fs::rename(path, &backup)
            .map_err(|error| format!("cannot back up PicPortal queue: {error}"))?;
    }
    if let Err(error) = fs::rename(&temporary, path) {
        if backup.exists() {
            let _ = fs::rename(&backup, path);
        }
        return Err(format!("cannot commit PicPortal queue: {error}"));
    }
    if let Ok(directory) = fs::File::open(parent) {
        let _ = directory.sync_all();
    }
    if backup.exists() {
        fs::remove_file(&backup)
            .map_err(|error| format!("cannot clear PicPortal queue backup: {error}"))?;
    }
    Ok(())
}

fn queue_identity_matches(
    item: &QueueItem,
    path: &str,
    session: &PicPortalSession,
    gallery_id: &str,
    source_sha256: &str,
) -> bool {
    item.path == path
        && item.api_base_url == session.base_url
        && item.account_id == session.account_id
        && item.gallery_id == gallery_id
        && item.source_sha256 == source_sha256
}

fn mark_queue_error(
    queue: &mut QueueFile,
    path: &str,
    session: &PicPortalSession,
    gallery_id: &str,
    error: &str,
) {
    if let Some(item) = queue.items.iter_mut().find(|item| {
        item.path == path
            && item.api_base_url == session.base_url
            && item.account_id == session.account_id
            && item.gallery_id == gallery_id
    }) {
        item.state = "failed".to_owned();
        item.last_error = Some(error.to_owned());
    }
}

fn upsert_queue(queue: &mut QueueFile, item: QueueItem) {
    if let Some(existing) = queue.items.iter_mut().find(|existing| {
        existing.path == item.path
            && existing.api_base_url == item.api_base_url
            && existing.account_id == item.account_id
            && existing.gallery_id == item.gallery_id
    }) {
        *existing = item;
    } else {
        queue.items.push(item);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processing_capabilities(face_processing_enabled: bool) -> ProcessingCapabilities {
        ProcessingCapabilities {
            derivatives: DerivativeCapabilities {
                tauri_enabled: true,
            },
            faces: FaceCapabilities {
                tauri_enabled: face_processing_enabled,
                model_id: "model".into(),
                model_digest: None,
                embedding_dim: 128,
                detector_input: 320,
                max_faces: 10,
                pipeline_version: "pipeline".into(),
            },
        }
    }

    fn gallery(face_filter_enabled: bool) -> GallerySummary {
        GallerySummary {
            id: "gallery".into(),
            title: "Gallery".into(),
            slug: "gallery".into(),
            photo_count: 0,
            face_filter_enabled,
        }
    }

    fn synthetic_session_for(account_id: &str) -> PicPortalSession {
        PicPortalSession {
            client: Client::new(),
            base_url: PRODUCTION_API_BASE_URL.to_owned(),
            account_id: account_id.to_owned(),
            admin: AdminIdentity {
                email: format!("{account_id}@example.test"),
                name: account_id.to_owned(),
            },
            cookies: Arc::new(Mutex::new(Vec::new())),
            persistent: false,
        }
    }

    fn synthetic_session() -> PicPortalSession {
        synthetic_session_for("synthetic-account")
    }

    #[test]
    fn failed_secure_replacement_keeps_the_previous_session_for_restart() {
        let previous = synthetic_session_for("account-a");
        let mut active = Some(previous.clone());
        let stored = stored_session_data(&previous).expect("stored account A");

        let result = persist_login_session(
            &mut active,
            synthetic_session_for("account-b"),
            |session| {
                assert_eq!(session.account_id, "account-b");
                Err("synthetic secure storage failure".to_owned())
            },
        );
        let error = match result {
            Ok(_) => panic!("login must fail when credential replacement fails"),
            Err(error) => error,
        };
        assert!(error.contains("login was not completed"));
        assert!(error.contains("synthetic secure storage failure"));
        assert_eq!(
            active.as_ref().map(|session| session.account_id.as_str()),
            Some("account-a")
        );

        let serialized = serde_json::to_string(&stored).expect("serialize prior credential");
        let restored: StoredSession =
            serde_json::from_str(&serialized).expect("restore prior credential");
        assert_eq!(restored.account_id, "account-a");
    }

    #[test]
    fn successful_secure_replacement_commits_the_new_session() {
        let previous = synthetic_session_for("account-a");
        let mut active = Some(previous.clone());
        let mut stored = stored_session_data(&previous).expect("stored account A");

        let committed =
            persist_login_session(&mut active, synthetic_session_for("account-b"), |session| {
                stored = stored_session_data(session)?;
                Ok(())
            })
            .expect("commit account B");

        assert_eq!(committed.account_id, "account-b");
        assert!(committed.persistent);
        assert_eq!(
            active.as_ref().map(|session| session.account_id.as_str()),
            Some("account-b")
        );
        assert_eq!(stored.account_id, "account-b");
    }

    #[test]
    fn direct_export_accepts_only_delivery_formats() {
        assert_eq!(
            normalize_picportal_output_format("jpeg").expect("jpeg"),
            "jpg"
        );
        assert_eq!(
            normalize_picportal_output_format("PNG").expect("png"),
            "png"
        );
        assert_eq!(
            normalize_picportal_output_format("webp").expect("webp"),
            "webp"
        );
        assert!(normalize_picportal_output_format("tiff").is_err());
        assert!(normalize_picportal_output_format("cube").is_err());
    }

    #[test]
    fn staged_exports_require_exactly_one_output_per_selection() {
        let directory = tempfile::tempdir().expect("staging directory");
        let first = directory.path().join("first.jpg");
        fs::write(&first, b"synthetic output").expect("first output");
        let staged = staged_exports(directory.path(), "jpg", 1).expect("one staged output");
        assert_eq!(staged, vec![first.to_string_lossy().to_string()]);

        fs::write(directory.path().join("second.jpg"), b"synthetic output").expect("second output");
        let error = staged_exports(directory.path(), "jpg", 1).expect_err("extra output");
        assert!(error.contains("produced 2 outputs"));
    }

    #[test]
    fn direct_export_identity_is_destination_and_consent_scoped() {
        let session = PicPortalSession {
            client: Client::new(),
            base_url: PRODUCTION_API_BASE_URL.to_owned(),
            account_id: "account".to_owned(),
            admin: AdminIdentity {
                email: "admin@example.test".to_owned(),
                name: "Admin".to_owned(),
            },
            cookies: Arc::new(Mutex::new(Vec::new())),
            persistent: false,
        };
        let settings = crate::export_processing::ExportSettings {
            jpeg_quality: 90,
            tiff_bit_depth: crate::export_processing::TiffBitDepth::Sixteen,
            resize: None,
            keep_metadata: true,
            preserve_timestamps: false,
            strip_gps: false,
            filename_template: Some("{original_filename}_edited".to_owned()),
            watermark: None,
            export_masks: false,
            preserve_folders: false,
            destination_type: Some("picportal".to_owned()),
            subfolder: None,
        };
        let paths = vec!["/synthetic/one.jpg".to_owned()];
        let no_faces = direct_export_operation_id(
            &session,
            "gallery",
            &paths,
            &["/synthetic".to_owned()],
            &settings,
            "jpg",
            &None,
            &None,
            false,
        )
        .expect("operation id");
        let with_faces = direct_export_operation_id(
            &session,
            "gallery",
            &paths,
            &["/synthetic".to_owned()],
            &settings,
            "jpg",
            &None,
            &None,
            true,
        )
        .expect("operation id");
        assert_ne!(no_faces, with_faces);
        assert_eq!(
            no_faces,
            direct_export_operation_id(
                &session,
                "gallery",
                &paths,
                &["/synthetic".to_owned()],
                &settings,
                "jpg",
                &None,
                &None,
                false,
            )
            .expect("stable operation id")
        );
    }

    #[test]
    fn secure_storage_failure_keeps_the_in_memory_session_for_logout_retry() {
        let state = PicPortalState::default();
        *state.session.lock().expect("session lock") = Some(synthetic_session());

        let result = complete_picportal_logout(
            &state,
            Err("synthetic secure storage failure".to_owned()),
            None,
        );

        assert!(
            result
                .expect_err("logout must fail")
                .contains("logout not completed")
        );
        assert!(state.session.lock().expect("session lock").is_some());
    }

    #[test]
    fn remote_logout_failure_is_reported_after_local_session_removal() {
        let state = PicPortalState::default();
        *state.session.lock().expect("session lock") = Some(synthetic_session());

        let result =
            complete_picportal_logout(&state, Ok(()), Some("synthetic network failure".to_owned()));

        assert!(
            result
                .expect_err("remote failure must remain visible")
                .contains("removed locally")
        );
        assert!(state.session.lock().expect("session lock").is_none());
    }

    #[test]
    fn only_one_picportal_publication_can_run_at_a_time() {
        let state = PicPortalState::default();
        let activity = begin_picportal_activity(&state).expect("first publication");
        assert!(begin_picportal_activity(&state).is_err());
        drop(activity);
        assert!(begin_picportal_activity(&state).is_ok());
    }

    #[test]
    fn gallery_creation_payload_uses_only_explicit_policy() {
        let (_, face_filter_enabled, event_payload) = create_gallery_payload(CreateGalleryInput {
            title: " Event ".into(),
            gallery_type: "event".into(),
            access_mode: "link".into(),
            status: "draft".into(),
            face_filter_enabled: false,
            client_name: Some("ignored client".into()),
            client_email: Some("ignored@example.com".into()),
            password: Some("ignored password".into()),
        })
        .expect("event gallery payload");
        assert!(!face_filter_enabled);
        assert_eq!(event_payload["clientName"], "");
        assert_eq!(event_payload["clientEmail"], "");
        assert_eq!(event_payload["password"], "");
        assert_eq!(event_payload["status"], "draft");

        let (_, face_filter_enabled, client_payload) = create_gallery_payload(CreateGalleryInput {
            title: "Client".into(),
            gallery_type: "client".into(),
            access_mode: "password".into(),
            status: "active".into(),
            face_filter_enabled: true,
            client_name: Some("Client Name".into()),
            client_email: Some("client@example.com".into()),
            password: Some("secret".into()),
        })
        .expect("client gallery payload");
        assert!(face_filter_enabled);
        assert_eq!(client_payload["clientName"], "Client Name");
        assert_eq!(client_payload["clientEmail"], "client@example.com");
        assert_eq!(client_payload["password"], "secret");

        let client_error = create_gallery_payload(CreateGalleryInput {
            title: "Client".into(),
            gallery_type: "client".into(),
            access_mode: "link".into(),
            status: "active".into(),
            face_filter_enabled: true,
            client_name: None,
            client_email: None,
            password: None,
        })
        .expect_err("client identity is required");
        assert_eq!(
            client_error,
            "client galleries require a client name and email"
        );

        let password_error = create_gallery_payload(CreateGalleryInput {
            title: "Client".into(),
            gallery_type: "client".into(),
            access_mode: "password".into(),
            status: "active".into(),
            face_filter_enabled: true,
            client_name: Some("Client Name".into()),
            client_email: Some("client@example.com".into()),
            password: Some(" ".into()),
        })
        .expect_err("password is required");
        assert_eq!(
            password_error,
            "password-protected galleries require a password"
        );
    }

    #[test]
    fn queue_destination_is_part_of_the_identity() {
        let mut queue = QueueFile::default();
        upsert_queue(
            &mut queue,
            QueueItem {
                path: "export.jpg".into(),
                api_base_url: PRODUCTION_API_BASE_URL.into(),
                account_id: "account-a".into(),
                gallery_id: "gallery-a".into(),
                source_sha256: "hash".into(),
                state: "uploaded".into(),
                photo_id: Some("photo".into()),
                face_analysis_uploaded: false,
                upload_id: None,
                upload_offset: 0,
                last_error: None,
            },
        );
        let item = queue
            .items
            .iter()
            .find(|item| item.path == "export.jpg")
            .expect("queue item");
        assert_eq!(item.gallery_id, "gallery-a");
        assert_ne!(item.gallery_id, "gallery-b");

        upsert_queue(
            &mut queue,
            QueueItem {
                path: "export.jpg".into(),
                api_base_url: PRODUCTION_API_BASE_URL.into(),
                account_id: "account-a".into(),
                gallery_id: "gallery-b".into(),
                source_sha256: "hash".into(),
                state: "queued".into(),
                photo_id: None,
                face_analysis_uploaded: false,
                upload_id: None,
                upload_offset: 0,
                last_error: None,
            },
        );
        assert_eq!(queue.items.len(), 2);
    }

    #[test]
    fn face_publication_requires_the_selected_gallery_to_enable_faces() {
        let capabilities = processing_capabilities(true);
        assert!(!face_publication_enabled(&gallery(false), &capabilities, false).expect("policy"));
        assert!(face_publication_enabled(&gallery(true), &capabilities, true).expect("policy"));
        assert!(
            face_publication_enabled(&gallery(true), &processing_capabilities(false), true)
                .is_err()
        );
        assert!(
            !face_publication_enabled(&gallery(true), &capabilities, false)
                .expect("explicit opt-out")
        );
        assert!(face_publication_enabled(&gallery(false), &capabilities, true).is_err());
    }

    #[test]
    fn completed_publication_runs_only_missing_authorized_face_phase() {
        let mut item = QueueItem {
            path: "export.jpg".into(),
            api_base_url: PRODUCTION_API_BASE_URL.into(),
            account_id: "account".into(),
            gallery_id: "gallery".into(),
            source_sha256: "source".into(),
            state: "uploaded".into(),
            photo_id: Some("photo".into()),
            face_analysis_uploaded: false,
            upload_id: None,
            upload_offset: 128,
            last_error: None,
        };

        assert_eq!(
            completed_face_analysis_photo(&item, true).expect("face phase"),
            Some("photo".into())
        );
        assert_eq!(
            completed_face_analysis_photo(&item, false).expect("gallery guard"),
            None
        );
        item.face_analysis_uploaded = true;
        assert_eq!(
            completed_face_analysis_photo(&item, true).expect("completed face phase"),
            None
        );
    }

    #[test]
    fn derivative_idempotency_key_is_a_stable_uuid_for_the_operation() {
        let first = idempotency_key(
            "https://example.test",
            "account",
            "gallery",
            "photo",
            "source",
        );
        let retry = idempotency_key(
            "https://example.test",
            "account",
            "gallery",
            "photo",
            "source",
        );
        let changed_source = idempotency_key(
            "https://example.test",
            "account",
            "gallery",
            "photo",
            "other-source",
        );
        let changed_account = idempotency_key(
            "https://example.test",
            "other-account",
            "gallery",
            "photo",
            "source",
        );
        let changed_destination = idempotency_key(
            "https://other.example.test",
            "account",
            "gallery",
            "photo",
            "source",
        );

        assert!(uuid::Uuid::parse_str(&first).is_ok());
        assert_eq!(first, retry);
        assert_ne!(first, changed_source);
        assert_ne!(first, changed_account);
        assert_ne!(first, changed_destination);
    }

    #[test]
    fn export_validation_rejects_an_oversized_source() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("export.jpg");
        let file = fs::File::create(&path).expect("export file");
        file.set_len(local_derivatives::MAX_SOURCE_BYTES + 1)
            .expect("oversized sparse export");

        let error = validate_export_path(&path).expect_err("oversized export must be rejected");
        assert_eq!(
            error,
            format!(
                "source must be between 1 and {} bytes",
                local_derivatives::MAX_SOURCE_BYTES
            )
        );
    }

    #[test]
    fn queue_round_trip_preserves_resumable_session_state() {
        let queue = QueueFile {
            items: vec![QueueItem {
                path: "export.jpg".into(),
                api_base_url: PRODUCTION_API_BASE_URL.into(),
                account_id: "account".into(),
                gallery_id: "gallery".into(),
                source_sha256: "source".into(),
                state: "uploading".into(),
                photo_id: None,
                face_analysis_uploaded: false,
                upload_id: Some("upload".into()),
                upload_offset: 8 * 1024 * 1024,
                last_error: None,
            }],
        };
        let encoded = serde_json::to_vec(&queue).expect("queue JSON");
        let restored: QueueFile = serde_json::from_slice(&encoded).expect("queue JSON");
        assert_eq!(restored.items[0].upload_id.as_deref(), Some("upload"));
        assert_eq!(restored.items[0].upload_offset, 8 * 1024 * 1024);
        assert_eq!(restored.items[0].gallery_id, "gallery");
    }

    #[test]
    fn reexport_replaces_the_obsolete_upload_identity() {
        let mut queue = QueueFile {
            items: vec![QueueItem {
                path: "export.jpg".into(),
                api_base_url: PRODUCTION_API_BASE_URL.into(),
                account_id: "account".into(),
                gallery_id: "gallery".into(),
                source_sha256: "old-source".into(),
                state: "uploading".into(),
                photo_id: None,
                face_analysis_uploaded: false,
                upload_id: Some("old-upload".into()),
                upload_offset: 42,
                last_error: None,
            }],
        };
        upsert_queue(
            &mut queue,
            QueueItem {
                path: "export.jpg".into(),
                api_base_url: PRODUCTION_API_BASE_URL.into(),
                account_id: "account".into(),
                gallery_id: "gallery".into(),
                source_sha256: "new-source".into(),
                state: "queued".into(),
                photo_id: None,
                face_analysis_uploaded: false,
                upload_id: None,
                upload_offset: 0,
                last_error: None,
            },
        );

        assert_eq!(queue.items.len(), 1);
        assert_eq!(queue.items[0].source_sha256, "new-source");
        assert_eq!(queue.items[0].upload_id, None);
        assert_eq!(queue.items[0].upload_offset, 0);
    }

    #[test]
    fn queue_load_recovers_a_flushed_temporary_commit() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("picportal-queue.json");
        let temporary = path.with_extension("json.tmp");
        let canonical_queue = QueueFile {
            items: vec![QueueItem {
                path: "export.jpg".into(),
                api_base_url: PRODUCTION_API_BASE_URL.into(),
                account_id: "account".into(),
                gallery_id: "gallery".into(),
                source_sha256: "source".into(),
                state: "uploading".into(),
                photo_id: None,
                face_analysis_uploaded: false,
                upload_id: Some("old-upload".into()),
                upload_offset: 64,
                last_error: None,
            }],
        };
        let temporary_queue = QueueFile {
            items: vec![QueueItem {
                upload_id: Some("new-upload".into()),
                upload_offset: 128,
                ..canonical_queue.items[0].clone()
            }],
        };
        fs::write(
            &path,
            serde_json::to_vec(&canonical_queue).expect("canonical queue JSON"),
        )
        .expect("canonical queue");
        fs::write(
            &temporary,
            serde_json::to_vec(&temporary_queue).expect("temporary queue JSON"),
        )
        .expect("temporary queue");

        let restored = load_queue(&path).expect("recover queue");
        assert_eq!(restored.items[0].upload_id.as_deref(), Some("new-upload"));
        assert_eq!(restored.items[0].upload_offset, 128);
        assert!(!temporary.exists());

        fs::write(&temporary, b"{").expect("interrupted next queue write");
        let restored_after_second_crash = load_queue(&path).expect("recover promoted queue");
        assert_eq!(
            restored_after_second_crash.items[0].upload_id.as_deref(),
            Some("new-upload")
        );
        assert_eq!(restored_after_second_crash.items[0].upload_offset, 128);
    }
}
