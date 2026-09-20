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
    sync::Mutex,
    time::Duration,
};

use crate::{face_processing, local_derivatives};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, State};

pub const PRODUCTION_API_BASE_URL: &str = "https://api.getpicportal.com";
const APP_ORIGIN: &str = "https://getpicportal.com";
const CHUNK_SIZE: usize = 8 * 1024 * 1024;
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
const DERIVATIVE_PIPELINE_VERSION: &str = "picportal-editor-local-derivatives-v1";

#[derive(Default)]
pub struct PicPortalState {
    pub session: Mutex<Option<PicPortalSession>>,
    pub face_runtime: Mutex<Option<face_processing::FaceRuntime>>,
}

#[derive(Clone)]
pub struct PicPortalSession {
    client: Client,
    base_url: String,
    account_id: String,
    admin: AdminIdentity,
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
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PublishResult {
    pub completed: usize,
    pub failed: usize,
    pub items: Vec<PublishItemResult>,
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

fn production_client() -> Result<Client, String> {
    Client::builder()
        .cookie_store(true)
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(5 * 60))
        .build()
        .map_err(|error| format!("cannot initialize PicPortal network client: {error}"))
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

#[tauri::command]
pub async fn picportal_login(
    email: String,
    password: String,
    state: State<'_, PicPortalState>,
) -> Result<PicPortalLoginResult, String> {
    if email.trim().is_empty() || password.is_empty() {
        return Err("PicPortal email and password are required".to_owned());
    }
    let client = production_client()?;
    let login_response = client
        .post(endpoint(PRODUCTION_API_BASE_URL, "auth/login"))
        .header("origin", APP_ORIGIN)
        .json(&serde_json::json!({ "email": email.trim(), "password": password }))
        .send()
        .await
        .map_err(|error| format!("PicPortal login failed: {error}"))?;
    let login_response = require_success(login_response).await?;
    let login = login_response
        .json::<LoginResponse>()
        .await
        .map_err(|error| format!("PicPortal login response is invalid: {error}"))?;

    let me_response = client
        .get(endpoint(PRODUCTION_API_BASE_URL, "admin/me"))
        .header("origin", APP_ORIGIN)
        .send()
        .await
        .map_err(|error| format!("PicPortal identity request failed: {error}"))?;
    let me = require_success(me_response)
        .await?
        .json::<MeResponse>()
        .await
        .map_err(|error| format!("PicPortal identity response is invalid: {error}"))?;
    let account_id = me
        .admin
        .admin_user_id
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "PicPortal identity did not include an account id".to_owned())?;
    let email = me
        .admin
        .email
        .or(login.admin.email)
        .unwrap_or_else(|| email.trim().to_owned());
    let name = me
        .admin
        .name
        .or(login.admin.name)
        .unwrap_or_else(|| email.clone());
    let session = PicPortalSession {
        client,
        base_url: PRODUCTION_API_BASE_URL.to_owned(),
        account_id,
        admin: AdminIdentity { email, name },
    };
    let galleries = session.galleries().await?;
    let result = PicPortalLoginResult {
        admin: session.admin.clone(),
        galleries,
    };
    *state
        .session
        .lock()
        .map_err(|_| "PicPortal session lock is poisoned".to_owned())? = Some(session);
    Ok(result)
}

#[tauri::command]
pub async fn picportal_logout(state: State<'_, PicPortalState>) -> Result<(), String> {
    let session = state
        .session
        .lock()
        .map_err(|_| "PicPortal session lock is poisoned".to_owned())?
        .clone();
    if let Some(session) = session {
        let response = session
            .client
            .post(endpoint(&session.base_url, "auth/logout"))
            .header("origin", APP_ORIGIN)
            .send()
            .await
            .map_err(|error| format!("PicPortal logout failed: {error}"))?;
        require_success(response).await?;
        let mut current = state
            .session
            .lock()
            .map_err(|_| "PicPortal session lock is poisoned".to_owned())?;
        if current.as_ref().is_some_and(|current| {
            current.base_url == session.base_url && current.account_id == session.account_id
        }) {
            current.take();
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn picportal_galleries(
    state: State<'_, PicPortalState>,
) -> Result<Vec<GallerySummary>, String> {
    let session = current_session(&state)?;
    session.galleries().await
}

#[tauri::command]
pub async fn picportal_create_gallery(
    input: CreateGalleryInput,
    state: State<'_, PicPortalState>,
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
        .map_err(|error| format!("gallery creation failed: {error}"))?;
    let created = require_success(response)
        .await?
        .json::<CreatedGalleryResponse>()
        .await
        .map_err(|error| format!("gallery creation response is invalid: {error}"))?
        .gallery;
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

#[tauri::command]
pub async fn picportal_publish(
    paths: Vec<String>,
    gallery_id: String,
    app: AppHandle,
    state: State<'_, PicPortalState>,
) -> Result<PublishResult, String> {
    if paths.is_empty() {
        return Err("select at least one exported image".to_owned());
    }
    let session = current_session(&state)?;
    let galleries = session.galleries().await?;
    let gallery = galleries
        .into_iter()
        .find(|gallery| gallery.id == gallery_id)
        .ok_or_else(|| "selected PicPortal gallery no longer exists".to_owned())?;
    let capabilities = session.processing_capabilities().await?;
    if !capabilities.derivatives.tauri_enabled {
        return Err("PicPortal local derivative processing is disabled by the server".to_owned());
    }
    let publish_faces = face_publication_enabled(&gallery, &capabilities)?;
    let queue_path = queue_path(&app)?;
    let mut queue = load_queue(&queue_path)?;
    let mut result = PublishResult {
        completed: 0,
        failed: 0,
        items: Vec::new(),
    };

    for path in paths {
        let item_result = publish_one(
            &session,
            &gallery,
            &capabilities,
            publish_faces,
            &path,
            &app,
            &state,
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
                result.failed += 1;
                mark_queue_error(&mut queue, &path, &session, &gallery.id, &error);
                result.items.push(PublishItemResult {
                    path,
                    state: "failed".to_owned(),
                    photo_id: None,
                    error: Some(error),
                });
            }
        }
    }
    save_queue(&queue_path, &queue)?;
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
        require_success(response)
            .await?
            .json()
            .await
            .map_err(|error| format!("processing capabilities response is invalid: {error}"))
    }
}

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
    let source_path = PathBuf::from(path);
    validate_export_path(&source_path)?;
    let source = fs::read(&source_path)
        .map_err(|error| format!("cannot read export {}: {error}", source_path.display()))?;
    let derivatives = local_derivatives::generate_local_derivatives(&source)?;
    let existing = queue.items.iter().find(|item| {
        queue_identity_matches(item, path, session, &gallery.id, &derivatives.source_sha256)
    });
    if let Some(existing) = existing {
        if existing.state == "uploaded" {
            return Ok(PublishItemResult {
                path: path.to_owned(),
                state: existing.state.clone(),
                photo_id: existing.photo_id.clone(),
                error: None,
            });
        }
    }
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
        let mut runtime = state
            .face_runtime
            .lock()
            .map_err(|_| "face runtime lock is poisoned".to_owned())?;
        let runtime = face_processing::ensure_runtime(&mut runtime, app)?;
        let analysis = runtime.analyze(&source)?;
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
        Some(analysis)
    } else {
        None
    };

    let filename = source_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("export.jpg");
    let content_type = content_type_for_path(&source_path)
        .ok_or_else(|| "unsupported exported image extension".to_owned())?;
    let photo_id = upload_original(
        session,
        &gallery.id,
        filename,
        content_type,
        &source,
        &derivatives.source_sha256,
        path,
        queue_path,
        queue,
    )
    .await?;
    let idempotency_key = idempotency_key(&gallery.id, &photo_id, &derivatives.source_sha256);
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
        let face_manifest = face_manifest(&analysis, &idempotency_key, source.len() as u64);
        let thumbnails = analysis
            .faces
            .iter()
            .map(|face| face.thumbnail.clone())
            .collect::<Vec<_>>();
        upload_face_analysis(session, &gallery.id, &photo_id, &face_manifest, &thumbnails).await?;
    }
    if let Some(item) = queue.items.iter_mut().find(|item| {
        queue_identity_matches(item, path, session, &gallery.id, &derivatives.source_sha256)
    }) {
        item.state = "uploaded".to_owned();
        item.photo_id = Some(photo_id.clone());
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
) -> Result<bool, String> {
    if !gallery.face_filter_enabled {
        return Ok(false);
    }
    if !capabilities.faces.tauri_enabled {
        return Err(
            "PicPortal local face processing is disabled by the server; no remote fallback was used"
                .to_owned(),
        );
    }
    Ok(true)
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

fn content_type_for_path(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => Some("image/jpeg"),
        "png" => Some("image/png"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

fn derivative_dimensions(derivative: &local_derivatives::LocalDerivative) -> DerivativeDimensions {
    DerivativeDimensions {
        sha256: derivative.sha256.clone(),
        width: derivative.width,
        height: derivative.height,
        bytes: derivative.bytes.len() as u64,
    }
}

fn idempotency_key(gallery_id: &str, photo_id: &str, source_sha256: &str) -> String {
    local_derivatives::sha256_hex(
        format!("{gallery_id}:{photo_id}:{source_sha256}:{DERIVATIVE_PIPELINE_VERSION}").as_bytes(),
    )
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
) -> Result<String, String> {
    let total_bytes = source.len() as u64;
    if let Some(existing) = queue
        .items
        .iter()
        .find(|item| queue_identity_matches(item, path, session, gallery_id, source_sha256))
    {
        if let Some(photo_id) = existing.photo_id.as_deref() {
            return Ok(photo_id.to_owned());
        }
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

    let upload_id = session_payload.upload_id.clone();
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
        assert!(!face_publication_enabled(&gallery(false), &capabilities).expect("policy"));
        assert!(face_publication_enabled(&gallery(true), &capabilities).expect("policy"));
        assert!(face_publication_enabled(&gallery(true), &processing_capabilities(false)).is_err());
    }

    #[test]
    fn idempotency_key_changes_with_source() {
        assert_ne!(
            idempotency_key("gallery", "photo", "a"),
            idempotency_key("gallery", "photo", "b")
        );
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
