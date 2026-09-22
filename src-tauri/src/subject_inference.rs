//! Local Grounding DINO, MediaPipe Pose, and VGG worker boundary.
//!
//! Python is an optional local adapter for the DINO and VGG checkpoints and
//! MediaPipe Pose runtime. The Rust culling pipeline enables each capability
//! only when its local model artifacts are present and verified. There is no
//! remote fallback.

use std::fs;
use std::io::{BufRead, BufReader, Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use image::{DynamicImage, ImageFormat};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Manager};

const CANONICAL_MANIFEST: &str = include_str!("../../models/culling/manifest.json");
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubjectBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub score: f32,
    pub label: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FocusMeasurement {
    pub score: f64,
    pub width: u32,
    pub height: u32,
    pub input_width: u32,
    pub input_height: u32,
    pub device: String,
}

#[derive(Debug, Clone)]
pub struct PoseBody {
    pub nose_x: f32,
    pub nose_y: f32,
    pub torso: Vec<(f32, f32)>,
}

#[derive(Debug, Clone)]
pub struct PoseFrame {
    pub width: f32,
    pub height: f32,
    pub bodies: Vec<PoseBody>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SubjectResponse {
    ok: bool,
    boxes: Option<Vec<SubjectBox>>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FocusResponse {
    ok: bool,
    score: Option<f64>,
    width: Option<u32>,
    height: Option<u32>,
    input_width: Option<u32>,
    input_height: Option<u32>,
    device: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PosePoint {
    x: f32,
    y: f32,
}

#[derive(Debug, Deserialize)]
struct PoseBodyWire {
    nose: PosePoint,
    torso: Vec<PosePoint>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PoseResponse {
    ok: bool,
    width: Option<u32>,
    height: Option<u32>,
    poses: Option<Vec<PoseBodyWire>>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CullingManifest {
    subject_model: SubjectModel,
    focus_model: FocusModel,
    pose_model: PoseModel,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PoseModel {
    filename: String,
    sha256: String,
    bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SubjectModel {
    license_filename: String,
    license_sha256: String,
    license_bytes: u64,
    artifacts: Vec<ModelArtifact>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FocusModel {
    checkpoint: String,
    bytes: u64,
    source_license_filename: String,
    source_license_sha256: String,
    source_license_bytes: u64,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct ModelArtifact {
    filename: String,
    sha256: String,
    bytes: u64,
}

#[derive(Clone)]
pub struct LocalWorkerCancellation {
    child: Arc<Mutex<Child>>,
}

impl LocalWorkerCancellation {
    pub fn cancel(&self) {
        let mut child = self.child.lock().unwrap_or_else(|error| error.into_inner());
        let _ = child.kill();
    }
}

pub struct LocalWorker {
    child: Arc<Mutex<Child>>,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    subject_ready: bool,
    focus_ready: bool,
    pose_ready: bool,
}

impl LocalWorker {
    pub fn start(app: &AppHandle) -> Result<Self, String> {
        let model_directory = model_directory(app)?;
        let subject_ready = verify_subject_bundle(&model_directory).is_ok();
        let script = worker_script_path(app)?;
        let python =
            std::env::var_os("PICPORTAL_CULLING_PYTHON").unwrap_or_else(|| "python3".into());
        let focus_ready = focus_model_ready(&model_directory).unwrap_or(false);
        let pose_ready = pose_model_ready(&model_directory);
        let mut child = Command::new(python)
            .arg(script)
            .arg("--local-worker")
            .env("PICPORTAL_CULLING_MODEL_DIR", &model_directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("LOCAL_CULLING_WORKER_START_FAILED: {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "LOCAL_CULLING_WORKER_STDIN_FAILED".to_owned())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "LOCAL_CULLING_WORKER_STDOUT_FAILED".to_owned())?;
        Ok(Self {
            child: Arc::new(Mutex::new(child)),
            stdin,
            stdout: BufReader::new(stdout),
            subject_ready,
            focus_ready,
            pose_ready,
        })
    }

    pub fn detect(
        &mut self,
        image: &DynamicImage,
        phrase: &str,
    ) -> Result<Vec<SubjectBox>, String> {
        if !self.subject_ready {
            return Err("LOCAL_CULLING_SUBJECT_MODEL_UNAVAILABLE".to_owned());
        }
        let payload = json!({
            "operation": "detect",
            "phrase": phrase,
            "image": encode_image(image, 1920)?,
        });
        let response: SubjectResponse = self.request(payload)?;
        if !response.ok {
            return Err(response
                .error
                .unwrap_or_else(|| "LOCAL_CULLING_SUBJECT_FAILED".to_owned()));
        }
        Ok(response.boxes.unwrap_or_default())
    }

    pub fn focus(&mut self, image: &DynamicImage) -> Result<FocusMeasurement, String> {
        if !self.focus_ready {
            return Err("LOCAL_CULLING_FOCUS_MODEL_UNAVAILABLE".to_owned());
        }
        let payload = json!({
            "operation": "focus",
            "image": encode_image(image, u32::MAX)?,
        });
        let response: FocusResponse = self.request(payload)?;
        if !response.ok {
            return Err(response
                .error
                .unwrap_or_else(|| "LOCAL_CULLING_FOCUS_FAILED".to_owned()));
        }
        Ok(FocusMeasurement {
            score: response
                .score
                .ok_or_else(|| "LOCAL_CULLING_FOCUS_SCORE_MISSING".to_owned())?,
            width: response
                .width
                .ok_or_else(|| "LOCAL_CULLING_FOCUS_WIDTH_MISSING".to_owned())?,
            height: response
                .height
                .ok_or_else(|| "LOCAL_CULLING_FOCUS_HEIGHT_MISSING".to_owned())?,
            input_width: response
                .input_width
                .ok_or_else(|| "LOCAL_CULLING_FOCUS_INPUT_WIDTH_MISSING".to_owned())?,
            input_height: response
                .input_height
                .ok_or_else(|| "LOCAL_CULLING_FOCUS_INPUT_HEIGHT_MISSING".to_owned())?,
            device: response.device.unwrap_or_else(|| "unknown".to_owned()),
        })
    }

    fn request<T: for<'de> Deserialize<'de>>(
        &mut self,
        payload: serde_json::Value,
    ) -> Result<T, String> {
        let line = serde_json::to_string(&payload)
            .map_err(|error| format!("LOCAL_CULLING_REQUEST_ENCODE_FAILED: {error}"))?;
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
            .map_err(|error| format!("LOCAL_CULLING_WORKER_WRITE_FAILED: {error}"))?;
        let mut response = String::new();
        self.stdout
            .read_line(&mut response)
            .map_err(|error| format!("LOCAL_CULLING_WORKER_READ_FAILED: {error}"))?;
        if response.len() > MAX_RESPONSE_BYTES {
            return Err("LOCAL_CULLING_WORKER_RESPONSE_TOO_LARGE".to_owned());
        }
        serde_json::from_str(&response)
            .map_err(|error| format!("LOCAL_CULLING_WORKER_RESPONSE_INVALID: {error}"))
    }

    pub fn pose(&mut self, image: &DynamicImage) -> Result<PoseFrame, String> {
        if !self.pose_ready {
            return Err("LOCAL_CULLING_POSE_MODEL_UNAVAILABLE".to_owned());
        }
        let payload = json!({
            "operation": "pose",
            "image": encode_image(image, 1920)?,
        });
        let response: PoseResponse = self.request(payload)?;
        if !response.ok {
            return Err(response
                .error
                .unwrap_or_else(|| "LOCAL_CULLING_POSE_FAILED".to_owned()));
        }
        let width = response
            .width
            .ok_or_else(|| "LOCAL_CULLING_POSE_WIDTH_MISSING".to_owned())?
            as f32;
        let height = response
            .height
            .ok_or_else(|| "LOCAL_CULLING_POSE_HEIGHT_MISSING".to_owned())?
            as f32;
        let bodies = response
            .poses
            .unwrap_or_default()
            .into_iter()
            .map(|pose| PoseBody {
                nose_x: pose.nose.x,
                nose_y: pose.nose.y,
                torso: pose
                    .torso
                    .into_iter()
                    .map(|point| (point.x, point.y))
                    .collect(),
            })
            .collect();
        Ok(PoseFrame {
            width,
            height,
            bodies,
        })
    }

    pub fn cancellation_handle(&self) -> LocalWorkerCancellation {
        LocalWorkerCancellation {
            child: Arc::clone(&self.child),
        }
    }

    pub fn subject_is_ready(&self) -> bool {
        self.subject_ready
    }

    pub fn focus_is_ready(&self) -> bool {
        self.focus_ready
    }

    pub fn pose_is_ready(&self) -> bool {
        self.pose_ready
    }
}

impl Drop for LocalWorker {
    fn drop(&mut self) {
        let mut child = self.child.lock().unwrap_or_else(|error| error.into_inner());
        let _ = child.kill();
        let _ = child.wait();
    }
}

pub fn model_directory(app: &AppHandle) -> Result<PathBuf, String> {
    let explicit = std::env::var_os("PICPORTAL_CULLING_MODEL_DIR").map(PathBuf::from);
    let packaged = app
        .path()
        .resolve("models/culling", tauri::path::BaseDirectory::Resource)
        .ok();
    select_model_directory(explicit, packaged, development_model_directory())
}

fn development_model_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/culling")
}

fn select_model_directory(
    explicit: Option<PathBuf>,
    packaged: Option<PathBuf>,
    development: PathBuf,
) -> Result<PathBuf, String> {
    if let Some(path) = explicit
        && path.join("manifest.json").is_file()
    {
        return Ok(path);
    }
    if let Some(path) = packaged
        && path.join("manifest.json").is_file()
    {
        return Ok(path);
    }
    if development.join("manifest.json").is_file() {
        return Ok(development);
    }
    Err("LOCAL_CULLING_MODEL_MANIFEST_UNAVAILABLE".to_owned())
}

fn worker_script_path(app: &AppHandle) -> Result<PathBuf, String> {
    if let Ok(path) = app.path().resolve(
        "scripts/culling_worker.py",
        tauri::path::BaseDirectory::Resource,
    ) && path.is_file()
    {
        return Ok(path);
    }
    let development =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../scripts/culling_worker.py");
    if development.is_file() {
        return Ok(development);
    }
    Err("LOCAL_CULLING_WORKER_UNAVAILABLE".to_owned())
}

fn verify_subject_bundle(model_directory: &Path) -> Result<(), String> {
    let manifest_bytes = fs::read(model_directory.join("manifest.json"))
        .map_err(|_| "LOCAL_CULLING_MANIFEST_INVALID".to_owned())?;
    let manifest: CullingManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|_| "LOCAL_CULLING_MANIFEST_INVALID".to_owned())?;
    let canonical: CullingManifest = serde_json::from_str(CANONICAL_MANIFEST)
        .map_err(|_| "LOCAL_CULLING_MANIFEST_INVALID".to_owned())?;
    if manifest.subject_model.license_filename != canonical.subject_model.license_filename
        || manifest.subject_model.license_sha256 != canonical.subject_model.license_sha256
        || manifest.subject_model.license_bytes != canonical.subject_model.license_bytes
        || manifest.subject_model.artifacts != canonical.subject_model.artifacts
        || manifest.focus_model.checkpoint != canonical.focus_model.checkpoint
        || manifest.focus_model.bytes != canonical.focus_model.bytes
        || manifest.focus_model.source_license_filename
            != canonical.focus_model.source_license_filename
        || manifest.focus_model.source_license_sha256 != canonical.focus_model.source_license_sha256
        || manifest.focus_model.source_license_bytes != canonical.focus_model.source_license_bytes
    {
        return Err("LOCAL_CULLING_MANIFEST_MISMATCH".to_owned());
    }
    let license_path = model_directory.join(&manifest.subject_model.license_filename);
    let license_metadata =
        fs::metadata(&license_path).map_err(|_| "LOCAL_CULLING_LICENSE_MISSING".to_owned())?;
    if license_metadata.len() != manifest.subject_model.license_bytes
        || sha256_file(&license_path)? != manifest.subject_model.license_sha256
    {
        return Err("LOCAL_CULLING_LICENSE_INVALID".to_owned());
    }
    for artifact in &manifest.subject_model.artifacts {
        let path = model_directory.join(&artifact.filename);
        let metadata = fs::metadata(&path)
            .map_err(|_| format!("LOCAL_CULLING_ARTIFACT_MISSING: {}", artifact.filename))?;
        if metadata.len() != artifact.bytes {
            return Err(format!(
                "LOCAL_CULLING_ARTIFACT_SIZE_INVALID: {}",
                artifact.filename
            ));
        }
        if sha256_file(&path)? != artifact.sha256 {
            return Err(format!(
                "LOCAL_CULLING_ARTIFACT_HASH_INVALID: {}",
                artifact.filename
            ));
        }
    }
    Ok(())
}

fn focus_model_ready(model_directory: &Path) -> Result<bool, String> {
    let manifest_bytes = fs::read(model_directory.join("manifest.json"))
        .map_err(|_| "LOCAL_CULLING_MANIFEST_INVALID".to_owned())?;
    let manifest: CullingManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|_| "LOCAL_CULLING_MANIFEST_INVALID".to_owned())?;
    let license_path = model_directory.join(&manifest.focus_model.source_license_filename);
    let license_metadata =
        fs::metadata(&license_path).map_err(|_| "LOCAL_CULLING_LICENSE_MISSING".to_owned())?;
    if license_metadata.len() != manifest.focus_model.source_license_bytes
        || sha256_file(&license_path)? != manifest.focus_model.source_license_sha256
    {
        return Err("LOCAL_CULLING_LICENSE_INVALID".to_owned());
    }
    let checkpoint = model_directory.join(&manifest.focus_model.checkpoint);
    let digest_path = checkpoint.with_extension("pth.sha256");
    if !checkpoint.is_file() || !digest_path.is_file() {
        return Ok(false);
    }
    let metadata = fs::metadata(&checkpoint).map_err(|error| error.to_string())?;
    if metadata.len() != manifest.focus_model.bytes {
        return Ok(false);
    }
    let expected = fs::read_to_string(digest_path)
        .map_err(|error| error.to_string())?
        .trim()
        .to_owned();
    Ok(!expected.is_empty() && sha256_file(&checkpoint)? == expected)
}

fn pose_model_ready(model_directory: &Path) -> bool {
    let Ok(manifest_bytes) = fs::read(model_directory.join("manifest.json")) else {
        return false;
    };
    let Ok(manifest) = serde_json::from_slice::<CullingManifest>(&manifest_bytes) else {
        return false;
    };
    let path = model_directory.join(&manifest.pose_model.filename);
    let Ok(metadata) = fs::metadata(&path) else {
        return false;
    };
    if metadata.len() != manifest.pose_model.bytes {
        return false;
    }
    sha256_file(&path)
        .ok()
        .is_some_and(|digest| digest == manifest.pose_model.sha256)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("LOCAL_CULLING_ARTIFACT_READ_FAILED: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(hex::encode(hasher.finalize()))
}

fn encode_image(image: &DynamicImage, max_dimension: u32) -> Result<String, String> {
    let prepared = if max_dimension == u32::MAX {
        image.clone()
    } else {
        image.thumbnail(max_dimension, max_dimension)
    };
    let mut output = Cursor::new(Vec::new());
    let format = if max_dimension == u32::MAX {
        ImageFormat::Png
    } else {
        ImageFormat::Jpeg
    };
    prepared
        .write_to(&mut output, format)
        .map_err(|error| format!("LOCAL_CULLING_IMAGE_ENCODE_FAILED: {error}"))?;
    Ok(BASE64.encode(output.into_inner()))
}

#[cfg(test)]
mod tests {
    use super::{
        CANONICAL_MANIFEST, CullingManifest, development_model_directory, select_model_directory,
    };
    use std::fs;
    use std::path::{Path, PathBuf};

    fn manifest_dir(path: &Path) -> PathBuf {
        fs::create_dir_all(path).unwrap();
        fs::write(path.join("manifest.json"), b"{}").unwrap();
        path.to_path_buf()
    }

    #[test]
    fn explicit_model_directory_wins_when_its_manifest_exists() {
        let root = tempfile::tempdir().unwrap();
        let explicit = manifest_dir(&root.path().join("explicit"));
        let packaged = manifest_dir(&root.path().join("packaged"));
        let development = manifest_dir(&root.path().join("development"));
        let stale_app_data =
            manifest_dir(&root.path().join("app-data").join("models").join("culling"));

        let selected =
            select_model_directory(Some(explicit.clone()), Some(packaged), development).unwrap();

        assert_eq!(selected, explicit);
        assert_ne!(selected, stale_app_data);
    }

    #[test]
    fn packaged_resource_wins_when_explicit_directory_has_no_manifest() {
        let root = tempfile::tempdir().unwrap();
        let explicit = root.path().join("explicit");
        fs::create_dir_all(&explicit).unwrap();
        let packaged = manifest_dir(&root.path().join("packaged"));
        let development = manifest_dir(&root.path().join("development"));

        let selected =
            select_model_directory(Some(explicit), Some(packaged.clone()), development).unwrap();

        assert_eq!(selected, packaged);
    }

    #[test]
    fn development_models_are_used_when_earlier_sources_have_no_manifest() {
        let root = tempfile::tempdir().unwrap();
        let development = manifest_dir(&root.path().join("development"));
        let stale_app_data =
            manifest_dir(&root.path().join("app-data").join("models").join("culling"));

        let selected = select_model_directory(
            None,
            Some(root.path().join("missing-resource")),
            development.clone(),
        )
        .unwrap();

        assert_eq!(selected, development);
        assert_ne!(selected, stale_app_data);
    }

    #[test]
    fn missing_manifests_stay_unavailable_instead_of_using_another_directory() {
        let root = tempfile::tempdir().unwrap();
        let stale_app_data =
            manifest_dir(&root.path().join("app-data").join("models").join("culling"));

        let error = select_model_directory(None, None, root.path().join("missing-development"))
            .unwrap_err();

        assert_eq!(error, "LOCAL_CULLING_MODEL_MANIFEST_UNAVAILABLE");
        assert!(stale_app_data.join("manifest.json").is_file());
    }

    #[test]
    fn development_fallback_resolves_the_repo_culling_models() {
        let selected = select_model_directory(None, None, development_model_directory()).unwrap();

        assert_eq!(
            selected.canonicalize().unwrap(),
            development_model_directory().canonicalize().unwrap()
        );
        assert!(selected.join("manifest.json").is_file());
    }

    #[test]
    fn shipped_manifest_loads_camel_case_licenses_and_sha256_digests() {
        let manifest: CullingManifest =
            serde_json::from_str(CANONICAL_MANIFEST).expect("shipped culling manifest");
        assert_eq!(manifest.subject_model.license_filename, "DINO-LICENSE.txt");
        assert_eq!(manifest.subject_model.license_bytes, 11355);
        assert_eq!(
            manifest.focus_model.source_license_filename,
            "VGG-LICENSE.txt"
        );
        assert_eq!(manifest.focus_model.checkpoint, "vgg_best.pth");
        assert_eq!(manifest.pose_model.filename, "pose_landmarker_lite.task");
        assert_eq!(manifest.pose_model.bytes, 5_777_746);
        assert_eq!(
            manifest.pose_model.sha256,
            "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
        );
        let config = manifest
            .subject_model
            .artifacts
            .iter()
            .find(|artifact| artifact.filename == "dino/config.json")
            .expect("config artifact");
        assert_eq!(config.bytes, 1644);
        assert_eq!(
            config.sha256,
            "eec82c5ab66e16df12a9a212e68ac011779927c2536cf9078658e35d85f0c67a"
        );
        for artifact in &manifest.subject_model.artifacts {
            assert_eq!(artifact.sha256.len(), 64, "{}", artifact.filename);
            assert!(
                artifact
                    .sha256
                    .chars()
                    .all(|character| character.is_ascii_hexdigit()),
                "{}",
                artifact.filename
            );
        }
    }
}
