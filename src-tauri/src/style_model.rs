//! Style model v1 ("Appliquer mon style"): proposes the basic sliders in the photographer's
//! style from features of the unedited image, with a small ONNX regressor trained by
//! `scripts/style/train.py`. Feature extraction mirrors `scripts/style/features.py`; see
//! docs/style-model.md for the contract.

use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use exif::{In, Reader as ExifReader, Tag};
use image::Rgb32FImage;
use ort::{session::Session, value::Tensor};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Manager};

use crate::file_management::{self, parse_virtual_path};
use crate::formats::is_raw_file;
use crate::image_loader;

pub const FEATURE_VERSION: &str = "style-features-v1";
pub const FEATURE_COUNT: usize = 55;
pub const OUTPUT_KEYS: [&str; 10] = [
    "exposure",
    "contrast",
    "highlights",
    "shadows",
    "whites",
    "blacks",
    "temperature",
    "tint",
    "vibrance",
    "saturation",
];
/// Adjustments key recording what the style wrote (and the values it replaced), so a later
/// run can tell its own values from manual edits.
pub const STYLE_MARKER_KEY: &str = "styleProvenance";

const PREVIEW_SIZE: usize = 64;
const GRID: usize = 4;
const PERCENTILES: [f64; 9] = [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0];
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_MODEL_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExifFeatures {
    pub iso: Option<f64>,
    pub exposure_time: Option<f64>,
    pub focal_length: Option<f64>,
}

fn exif_rational(exif: &exif::Exif, tag: Tag) -> Option<f64> {
    match &exif.get_field(tag, In::PRIMARY)?.value {
        exif::Value::Rational(values) => values
            .first()
            .filter(|value| value.denom != 0)
            .map(|value| value.to_f64()),
        exif::Value::Double(values) => values.first().copied(),
        exif::Value::Float(values) => values.first().map(|value| f64::from(*value)),
        _ => None,
    }
}

pub fn read_exif_features(bytes: &[u8]) -> ExifFeatures {
    let Ok(exif) = ExifReader::new().read_from_container(&mut Cursor::new(bytes)) else {
        return ExifFeatures::default();
    };
    ExifFeatures {
        iso: exif
            .get_field(Tag::PhotographicSensitivity, In::PRIMARY)
            .and_then(|field| field.value.get_uint(0))
            .map(f64::from),
        exposure_time: exif_rational(&exif, Tag::ExposureTime),
        focal_length: exif_rational(&exif, Tag::FocalLength),
    }
}

/// Display-referred sRGB source used for the features: the decoded file, or the largest
/// embedded JPEG preview for RAW files (same rule as the Python extractor).
pub fn decode_feature_source(path: &Path, bytes: &[u8]) -> Result<Rgb32FImage, String> {
    let path_str = path.to_string_lossy();
    if is_raw_file(path) {
        image_loader::safe_embedded_preview_fallback(bytes, &path_str)
            .map(|preview| preview.to_rgb32f())
            .ok_or_else(|| "STYLE_RAW_PREVIEW_UNAVAILABLE".to_owned())
    } else {
        image_loader::load_image_with_orientation(bytes, None)
            .map(|image| image.to_rgb32f())
            .map_err(|error| format!("STYLE_IMAGE_DECODE_FAILED: {error}"))
    }
}

fn block_bounds(length: usize, cells: usize) -> Vec<(usize, usize)> {
    let starts: Vec<usize> = (0..cells)
        .map(|index| ((index * length) / cells).min(length - 1))
        .collect();
    (0..cells)
        .map(|index| {
            let end = if index + 1 < cells {
                starts[index + 1]
            } else {
                length
            };
            (starts[index], end.max(starts[index] + 1))
        })
        .collect()
}

/// Mean colour of PREVIEW_SIZE x PREVIEW_SIZE integer-bounded blocks, row-major.
pub fn box_preview(image: &Rgb32FImage) -> Vec<[f64; 3]> {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let rows = block_bounds(height, PREVIEW_SIZE);
    let columns = block_bounds(width, PREVIEW_SIZE);
    let mut preview = Vec::with_capacity(PREVIEW_SIZE * PREVIEW_SIZE);
    for &(top, bottom) in &rows {
        for &(left, right) in &columns {
            let mut sum = [0.0_f64; 3];
            for y in top..bottom {
                for x in left..right {
                    let pixel = image.get_pixel(x as u32, y as u32);
                    for channel in 0..3 {
                        sum[channel] += f64::from(pixel[channel]);
                    }
                }
            }
            let count = ((bottom - top) * (right - left)) as f64;
            preview.push([sum[0] / count, sum[1] / count, sum[2] / count]);
        }
    }
    preview
}

fn log2_or(value: Option<f64>, default: f64) -> f64 {
    match value {
        Some(value) if value > 0.0 => value.log2(),
        _ => default.log2(),
    }
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values.fold((0.0, 0usize), |(sum, count), value| (sum + value, count + 1));
    if count == 0 { 0.0 } else { sum / count as f64 }
}

pub fn compute_features(preview: &[[f64; 3]], exif: &ExifFeatures) -> Vec<f32> {
    let luma: Vec<f64> = preview
        .iter()
        .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
        .collect();
    let warmth: Vec<f64> = preview.iter().map(|p| p[0] - p[2]).collect();
    let mean_luma = mean(luma.iter().copied());
    let std_luma = mean(luma.iter().map(|y| (y - mean_luma).powi(2))).sqrt();
    let mut sorted = luma.clone();
    sorted.sort_by(f64::total_cmp);
    let last = (sorted.len() - 1) as f64;

    let mut features = vec![
        mean(preview.iter().map(|p| p[0])),
        mean(preview.iter().map(|p| p[1])),
        mean(preview.iter().map(|p| p[2])),
        mean_luma,
        std_luma,
    ];
    features.extend(
        PERCENTILES
            .iter()
            .map(|p| sorted[(p / 100.0 * last + 0.5).floor() as usize]),
    );
    features.push(mean(luma.iter().map(|y| f64::from(u8::from(*y >= 0.98)))));
    features.push(mean(luma.iter().map(|y| f64::from(u8::from(*y <= 0.02)))));
    features.push(mean(warmth.iter().copied()));
    features.push(mean(preview.iter().map(|p| p[1] - (p[0] + p[2]) / 2.0)));
    features.push(mean(preview.iter().map(|p| {
        p[0].max(p[1]).max(p[2]) - p[0].min(p[1]).min(p[2])
    })));
    let cell = PREVIEW_SIZE / GRID;
    for values in [&luma, &warmth] {
        for row in 0..GRID {
            for column in 0..GRID {
                features.push(mean((0..cell * cell).map(|index| {
                    let y = row * cell + index / cell;
                    let x = column * cell + index % cell;
                    values[y * PREVIEW_SIZE + x]
                })));
            }
        }
    }
    let present = exif.iso.is_some() && exif.exposure_time.is_some() && exif.focal_length.is_some();
    features.push(f64::from(u8::from(present)));
    features.push(log2_or(exif.iso, 100.0) - 100.0_f64.log2());
    features.push(log2_or(exif.exposure_time, 1.0 / 125.0));
    features.push(log2_or(exif.focal_length, 50.0));
    debug_assert_eq!(features.len(), FEATURE_COUNT);
    features.into_iter().map(|value| value as f32).collect()
}

pub fn features_for_source(path: &Path, bytes: &[u8]) -> Result<Vec<f32>, String> {
    let image = decode_feature_source(path, bytes)?;
    Ok(compute_features(&box_preview(&image), &read_exif_features(bytes)))
}

// --- Model bundle ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StyleArtifact {
    pub role: String,
    pub filename: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StyleManifest {
    pub version: u32,
    pub model_id: String,
    pub feature_version: String,
    pub feature_count: usize,
    pub output_keys: Vec<String>,
    pub kind: String,
    pub fallback: bool,
    pub training_samples: u32,
    pub minimum_for_learned_model: u32,
    pub artifacts: Vec<StyleArtifact>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StyleModelInfo {
    pub model_id: String,
    pub kind: String,
    pub fallback: bool,
    pub training_samples: u32,
    pub minimum_for_learned_model: u32,
}

pub struct StyleRuntime {
    manifest_bytes: Vec<u8>,
    directory: PathBuf,
    pub manifest: StyleManifest,
    session: Session,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

pub fn load_verified_manifest(directory: &Path) -> Result<(StyleManifest, Vec<u8>, PathBuf), String> {
    let manifest_path = directory.join("manifest.json");
    let metadata =
        fs::metadata(&manifest_path).map_err(|_| "STYLE_MODEL_UNAVAILABLE".to_owned())?;
    if !metadata.is_file() || metadata.len() > MAX_MANIFEST_BYTES {
        return Err("STYLE_MODEL_MANIFEST_INVALID".to_owned());
    }
    let manifest_bytes =
        fs::read(&manifest_path).map_err(|_| "STYLE_MODEL_MANIFEST_INVALID".to_owned())?;
    let manifest: StyleManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("STYLE_MODEL_MANIFEST_INVALID: {error}"))?;
    if manifest.version != 1
        || manifest.feature_version != FEATURE_VERSION
        || manifest.feature_count != FEATURE_COUNT
        || manifest.output_keys != OUTPUT_KEYS
    {
        return Err("STYLE_MODEL_CONTRACT_MISMATCH".to_owned());
    }
    let [artifact] = manifest.artifacts.as_slice() else {
        return Err("STYLE_MODEL_MANIFEST_INVALID".to_owned());
    };
    if artifact.role != "regressor"
        || Path::new(&artifact.filename).file_name() != Some(artifact.filename.as_ref())
        || artifact.bytes > MAX_MODEL_BYTES
    {
        return Err("STYLE_MODEL_MANIFEST_INVALID".to_owned());
    }
    let model_path = directory.join(&artifact.filename);
    let model_bytes = fs::read(&model_path).map_err(|_| "STYLE_MODEL_ARTIFACT_MISSING".to_owned())?;
    if model_bytes.len() as u64 != artifact.bytes || sha256_hex(&model_bytes) != artifact.sha256 {
        return Err("STYLE_MODEL_ARTIFACT_HASH_MISMATCH".to_owned());
    }
    Ok((manifest, manifest_bytes, model_path))
}

impl StyleRuntime {
    pub fn load(directory: &Path) -> Result<Self, String> {
        let (manifest, manifest_bytes, model_path) = load_verified_manifest(directory)?;
        let _ = ort::init().with_name("PicPortal Style").commit();
        let session = Session::builder()
            .map_err(|error| format!("STYLE_RUNTIME_INIT_FAILED: {error}"))?
            .commit_from_file(model_path)
            .map_err(|error| format!("STYLE_MODEL_LOAD_FAILED: {error}"))?;
        Ok(Self {
            manifest_bytes,
            directory: directory.to_path_buf(),
            manifest,
            session,
        })
    }

    pub fn info(&self) -> StyleModelInfo {
        StyleModelInfo {
            model_id: self.manifest.model_id.clone(),
            kind: self.manifest.kind.clone(),
            fallback: self.manifest.fallback,
            training_samples: self.manifest.training_samples,
            minimum_for_learned_model: self.manifest.minimum_for_learned_model,
        }
    }

    /// Raw model output in slider units (not clamped).
    pub fn predict_raw(&mut self, features: &[f32]) -> Result<Vec<f32>, String> {
        if features.len() != FEATURE_COUNT {
            return Err("STYLE_FEATURES_INVALID".to_owned());
        }
        let tensor = Tensor::from_array(([1_usize, FEATURE_COUNT], features.to_vec()))
            .map_err(|error| format!("STYLE_INPUT_FAILED: {error}"))?;
        let outputs = self
            .session
            .run(ort::inputs![tensor])
            .map_err(|error| format!("STYLE_INFERENCE_FAILED: {error}"))?;
        let values: Vec<f32> = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|error| format!("STYLE_OUTPUT_FAILED: {error}"))?
            .iter()
            .copied()
            .collect();
        if values.len() != OUTPUT_KEYS.len() || values.iter().any(|value| !value.is_finite()) {
            return Err("STYLE_OUTPUT_FAILED".to_owned());
        }
        Ok(values)
    }

    /// Proposal clamped to the slider ranges and rounded like the sliders.
    pub fn predict(&mut self, features: &[f32]) -> Result<Map<String, Value>, String> {
        Ok(to_proposal(&self.predict_raw(features)?))
    }
}

pub fn to_proposal(values: &[f32]) -> Map<String, Value> {
    OUTPUT_KEYS
        .iter()
        .zip(values)
        .map(|(key, value)| {
            let value = f64::from(*value);
            let rounded = if *key == "exposure" {
                (value.clamp(-5.0, 5.0) * 100.0).round() / 100.0
            } else {
                value.clamp(-100.0, 100.0).round()
            };
            ((*key).to_owned(), json!(rounded))
        })
        .collect()
}

static RUNTIME: Mutex<Option<StyleRuntime>> = Mutex::new(None);

/// User-trained bundle first (app data), then the bundled resource, then the source tree.
pub fn model_directories(app_handle: &AppHandle) -> Vec<PathBuf> {
    let mut directories = Vec::new();
    if let Ok(data) = app_handle.path().app_data_dir() {
        directories.push(data.join("models/style"));
    }
    if let Ok(resource) = app_handle
        .path()
        .resolve("models/style", tauri::path::BaseDirectory::Resource)
    {
        directories.push(resource);
    }
    directories.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/style"));
    directories
}

/// Run `operation` with the current model, (re)loading it when the manifest changed.
pub fn with_runtime<T>(
    app_handle: &AppHandle,
    operation: impl FnOnce(&mut StyleRuntime) -> Result<T, String>,
) -> Result<T, String> {
    let directory = model_directories(app_handle)
        .into_iter()
        .find(|directory| directory.join("manifest.json").is_file())
        .ok_or_else(|| "STYLE_MODEL_UNAVAILABLE".to_owned())?;
    let manifest_bytes = fs::read(directory.join("manifest.json"))
        .map_err(|_| "STYLE_MODEL_UNAVAILABLE".to_owned())?;
    let mut guard = RUNTIME.lock().unwrap_or_else(|error| error.into_inner());
    let stale = guard.as_ref().is_none_or(|runtime| {
        runtime.directory != directory || runtime.manifest_bytes != manifest_bytes
    });
    if stale {
        *guard = None;
        *guard = Some(StyleRuntime::load(&directory)?);
    }
    operation(guard.as_mut().expect("runtime loaded above"))
}

// --- Non-destructive merge ------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StyleMergePolicy {
    /// Editor: explicit gesture on one photo, every key is written (undo restores).
    Overwrite,
    /// Batch: photos whose basic sliders were set by hand are left untouched.
    SkipEdited,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StyleMergeOutcome {
    pub applied: Vec<String>,
    /// Keys holding a manual value (not default, not the style's last value).
    pub manual: Vec<String>,
    pub skipped: bool,
}

fn provenance_value(provenance: Option<&Value>, field: &str, key: &str) -> Option<f64> {
    provenance?.get(field)?.get(key)?.as_f64()
}

/// A key belongs to the style when it is still at its default (0) or still holds the value
/// the style wrote (recorded in `styleProvenance`); anything else is a manual edit.
fn is_style_owned(map: &Map<String, Value>, key: &str) -> bool {
    let current = map.get(key).and_then(Value::as_f64).unwrap_or(0.0);
    current == 0.0
        || provenance_value(map.get(STYLE_MARKER_KEY), "values", key)
            .is_some_and(|value| (value - current).abs() < 1e-9)
}

/// Write the proposal into `adjustments` and record `styleProvenance`
/// (`modelId`, `appliedAt`, `keys`, `values` written, `previous` values to restore).
pub fn merge_style_proposal(
    adjustments: &mut Value,
    proposal: &Map<String, Value>,
    model_id: &str,
    applied_at: &str,
    policy: StyleMergePolicy,
) -> StyleMergeOutcome {
    if !adjustments.is_object() {
        *adjustments = json!({});
    }
    let map = adjustments.as_object_mut().expect("object ensured above");
    let mut outcome = StyleMergeOutcome {
        manual: OUTPUT_KEYS
            .iter()
            .filter(|key| !is_style_owned(map, key))
            .map(|key| (*key).to_owned())
            .collect(),
        ..StyleMergeOutcome::default()
    };
    if policy == StyleMergePolicy::SkipEdited && !outcome.manual.is_empty() {
        outcome.skipped = true;
        return outcome;
    }
    let old_provenance = map.get(STYLE_MARKER_KEY).cloned();
    let mut previous = Map::new();
    let mut values = Map::new();
    for key in OUTPUT_KEYS {
        let Some(proposed) = proposal.get(key) else {
            continue;
        };
        let current = map.get(key).and_then(Value::as_f64).unwrap_or(0.0);
        let written_by_style = provenance_value(old_provenance.as_ref(), "values", key)
            .is_some_and(|value| (value - current).abs() < 1e-9);
        // Re-applying keeps the value from before the first application.
        let before = if written_by_style {
            provenance_value(old_provenance.as_ref(), "previous", key).unwrap_or(0.0)
        } else {
            current
        };
        previous.insert(key.to_owned(), json!(before));
        values.insert(key.to_owned(), proposed.clone());
        map.insert(key.to_owned(), proposed.clone());
        outcome.applied.push(key.to_owned());
    }
    map.insert(
        STYLE_MARKER_KEY.to_owned(),
        json!({
            "modelId": model_id,
            "appliedAt": applied_at,
            "keys": outcome.applied,
            "values": values,
            "previous": previous,
        }),
    );
    outcome
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

// --- Commands -------------------------------------------------------------------------------

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StyleEditorProposal {
    pub model: StyleModelInfo,
    /// Keys to spread over the editor's adjustments (the 10 sliders and `styleProvenance`).
    pub patch: Map<String, Value>,
    pub outcome: StyleMergeOutcome,
}

/// Editor panel: propose the style for the open image without touching the sidecar; the
/// frontend applies `patch` as one undoable adjustment change.
#[tauri::command]
pub async fn calculate_style_adjustments(
    path: String,
    current_adjustments: Value,
    app_handle: AppHandle,
) -> Result<StyleEditorProposal, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let (source_path, _) = parse_virtual_path(&path);
        let bytes = fs::read(&source_path).map_err(|error| format!("STYLE_SOURCE_READ_FAILED: {error}"))?;
        let features = features_for_source(&source_path, &bytes)?;
        let (model, proposal) =
            with_runtime(&app_handle, |runtime| Ok((runtime.info(), runtime.predict(&features)?)))?;
        let mut merged = current_adjustments;
        let outcome = merge_style_proposal(
            &mut merged,
            &proposal,
            &model.model_id,
            &now_rfc3339(),
            StyleMergePolicy::Overwrite,
        );
        let mut patch = Map::new();
        for key in outcome.applied.iter().map(String::as_str).chain([STYLE_MARKER_KEY]) {
            patch.insert(key.to_owned(), merged[key].clone());
        }
        Ok(StyleEditorProposal { model, patch, outcome })
    })
    .await
    .map_err(|error| format!("STYLE_TASK_FAILED: {error}"))?
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StyleApplySummary {
    pub model: StyleModelInfo,
    pub updated: usize,
    pub skipped: usize,
    pub failed: usize,
    /// Error codes of the first failures (no file paths).
    pub errors: Vec<String>,
}

/// Library: write the proposal into the sidecars of `paths` through the same path as
/// `apply_auto_adjustments_to_paths` (sidecar lock, XMP sync, thumbnail refresh). With
/// `skip_edited` (default), photos whose basic sliders were set by hand are not modified.
#[tauri::command]
pub async fn apply_style_to_paths(
    paths: Vec<String>,
    skip_edited: Option<bool>,
    app_handle: AppHandle,
) -> Result<StyleApplySummary, String> {
    let model = with_runtime(&app_handle, |runtime| Ok(runtime.info()))?;
    let policy = if skip_edited.unwrap_or(true) {
        StyleMergePolicy::SkipEdited
    } else {
        StyleMergePolicy::Overwrite
    };
    let state = app_handle.state::<crate::AppState>();
    file_management::add_to_thumbnail_queue(&state, paths.len(), &app_handle);
    let model_id = model.model_id.clone();
    let applied_at = now_rfc3339();
    let task_handle = app_handle.clone();
    let results = tauri::async_runtime::spawn_blocking(move || {
        let results = file_management::update_adjustments_for_paths(
            &paths,
            &task_handle,
            |source_path, bytes, _settings| {
                let features = features_for_source(source_path, bytes)?;
                let proposal = with_runtime(&task_handle, |runtime| runtime.predict(&features))?;
                Ok((None, Value::Object(proposal)))
            },
            |adjustments, proposal| {
                let proposal = proposal.as_object().cloned().unwrap_or_default();
                let outcome =
                    merge_style_proposal(adjustments, &proposal, &model_id, &applied_at, policy);
                let write = !outcome.skipped;
                (outcome, write)
            },
        );
        paths.into_iter().zip(results).collect::<Vec<_>>()
    })
    .await
    .map_err(|error| format!("STYLE_TASK_FAILED: {error}"))?;

    let mut summary = StyleApplySummary {
        model,
        updated: 0,
        skipped: 0,
        failed: 0,
        errors: Vec::new(),
    };
    for (path, result) in results {
        match result {
            Ok(outcome) if outcome.skipped => summary.skipped += 1,
            Ok(_) => summary.updated += 1,
            Err(error) => {
                log::warn!("Failed to apply style to {}: {}", path, error);
                summary.failed += 1;
                if summary.errors.len() < 5 {
                    let code = error.split(':').next().unwrap_or_default().to_owned();
                    summary.errors.push(code);
                }
            }
        }
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/style/fixtures")
    }

    fn read_json(path: &Path) -> Value {
        serde_json::from_slice(&fs::read(path).expect("fixture")).expect("fixture JSON")
    }

    fn use_bundled_onnxruntime() {
        let name = if cfg!(target_os = "windows") {
            "onnxruntime.dll"
        } else if cfg!(target_os = "macos") {
            "libonnxruntime.dylib"
        } else {
            "libonnxruntime.so"
        };
        let library = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("resources/{name}"));
        if library.is_file() {
            unsafe { std::env::set_var("ORT_DYLIB_PATH", &library) };
        }
    }

    #[test]
    fn block_bounds_match_the_python_extractor() {
        assert_eq!(block_bounds(4, 4), vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
        assert_eq!(block_bounds(2, 4), vec![(0, 1), (0, 1), (1, 2), (1, 2)]);
        assert_eq!(block_bounds(10, 4), vec![(0, 2), (2, 5), (5, 7), (7, 10)]);
    }

    #[test]
    fn features_match_the_python_reference_on_fixtures() {
        assert_features_match(&fixtures().join("parity.expected.json"), &fixtures());
    }

    /// Opt-in parity on a local corpus: `STYLE_PARITY_EXPECTED` names a JSON written by the
    /// Python extractor (same format as parity.expected.json, `file` relative to the JSON's folder
    /// or absolute). Nothing of the corpus is stored in the repository.
    #[test]
    fn features_match_the_python_reference_on_a_local_corpus() {
        let Some(expected) = std::env::var_os("STYLE_PARITY_EXPECTED").map(PathBuf::from) else {
            return;
        };
        let base = expected.parent().expect("parent").to_path_buf();
        assert_features_match(&expected, &base);
    }

    fn assert_features_match(expected: &Path, base: &Path) {
        let expected = read_json(expected);
        for entry in expected["images"].as_array().expect("images") {
            let name = entry["file"].as_str().expect("file");
            let tolerance = entry["tolerance"].as_f64().expect("tolerance");
            let path = base.join(name);
            let bytes = fs::read(&path).expect("fixture image");
            let features = features_for_source(&path, &bytes).expect("features");
            let reference: Vec<f64> = entry["features"]
                .as_array()
                .expect("features")
                .iter()
                .map(|value| value.as_f64().expect("number"))
                .collect();
            assert_eq!(features.len(), reference.len());
            for (index, (actual, wanted)) in features.iter().zip(&reference).enumerate() {
                assert!(
                    (f64::from(*actual) - wanted).abs() <= tolerance,
                    "{name}: feature {index} is {actual}, Python computed {wanted}"
                );
            }
        }
    }

    #[test]
    fn fixture_model_reproduces_the_python_predictions() {
        use_bundled_onnxruntime();
        let expected = read_json(&fixtures().join("parity.expected.json"));
        let mut runtime = StyleRuntime::load(&fixtures().join("synthetic-model")).expect("runtime");
        assert!(!runtime.manifest.fallback);
        for entry in expected["images"].as_array().expect("images") {
            let features: Vec<f32> = entry["features"]
                .as_array()
                .expect("features")
                .iter()
                .map(|value| value.as_f64().expect("number") as f32)
                .collect();
            let output = runtime.predict_raw(&features).expect("prediction");
            for (actual, wanted) in output.iter().zip(entry["prediction"].as_array().expect("prediction")) {
                let wanted = wanted.as_f64().expect("number");
                assert!(
                    (f64::from(*actual) - wanted).abs() <= 1e-3 * wanted.abs().max(1.0),
                    "ONNX output {actual}, Python {wanted}"
                );
            }
        }
    }

    #[test]
    fn tampered_model_is_rejected() {
        let directory = std::env::temp_dir().join(format!("style-tamper-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("temp dir");
        let source = fixtures().join("synthetic-model");
        fs::copy(source.join("manifest.json"), directory.join("manifest.json")).expect("copy");
        let mut model = fs::read(source.join("style-model.onnx")).expect("model");
        let last = model.len() - 1;
        model[last] ^= 0xff;
        fs::write(directory.join("style-model.onnx"), model).expect("write");
        let error = load_verified_manifest(&directory).err().expect("must fail");
        assert_eq!(error, "STYLE_MODEL_ARTIFACT_HASH_MISMATCH");
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn proposal_is_clamped_and_rounded_like_the_sliders() {
        let proposal = to_proposal(&[7.123, 12.4, -250.0, 0.5, -0.4, 3.0, 99.6, -1.0, 0.0, 1.0]);
        assert_eq!(proposal["exposure"], json!(5.0));
        assert_eq!(proposal["contrast"], json!(12.0));
        assert_eq!(proposal["highlights"], json!(-100.0));
        assert_eq!(proposal["temperature"], json!(100.0));
    }

    fn sample_proposal(exposure: f32, highlights: f32) -> Map<String, Value> {
        to_proposal(&[exposure, 20.0, highlights, 10.0, 5.0, -5.0, 12.0, 3.0, 15.0, -4.0])
    }

    #[test]
    fn batch_skips_photos_with_manual_basic_settings() {
        let mut adjustments = json!({ "contrast": 42, "clarity": 7 });
        let before = adjustments.clone();
        let outcome = merge_style_proposal(
            &mut adjustments,
            &sample_proposal(0.5, -30.0),
            "m1",
            "t1",
            StyleMergePolicy::SkipEdited,
        );
        assert!(outcome.skipped);
        assert_eq!(outcome.manual, vec!["contrast".to_owned()]);
        assert_eq!(adjustments, before);
    }

    #[test]
    fn style_values_stay_replaceable_and_remember_the_original_values() {
        let mut adjustments = json!({ "exposure": 0, "clarity": 7 });
        let first = merge_style_proposal(
            &mut adjustments,
            &sample_proposal(0.5, -30.0),
            "m1",
            "t1",
            StyleMergePolicy::SkipEdited,
        );
        assert!(!first.skipped);
        assert_eq!(first.applied.len(), 10);
        assert_eq!(adjustments["exposure"], json!(0.5));
        assert_eq!(adjustments["clarity"], json!(7));

        // A second batch run treats the style's own values as replaceable ...
        let second = merge_style_proposal(
            &mut adjustments,
            &sample_proposal(0.8, -35.0),
            "m2",
            "t2",
            StyleMergePolicy::SkipEdited,
        );
        assert!(!second.skipped);
        assert_eq!(adjustments["exposure"], json!(0.8));
        let provenance = &adjustments[STYLE_MARKER_KEY];
        assert_eq!(provenance["modelId"], json!("m2"));
        assert_eq!(provenance["previous"]["exposure"], json!(0.0));
        assert_eq!(provenance["values"]["highlights"], json!(-35.0));

        // ... but a slider moved by hand makes the photo "already edited".
        adjustments["shadows"] = json!(33.0);
        let third = merge_style_proposal(
            &mut adjustments,
            &sample_proposal(1.0, -40.0),
            "m3",
            "t3",
            StyleMergePolicy::SkipEdited,
        );
        assert!(third.skipped);
        assert_eq!(third.manual, vec!["shadows".to_owned()]);
        assert_eq!(adjustments["exposure"], json!(0.8));
    }

    #[test]
    fn editor_overwrite_records_manual_values_for_restoration() {
        let mut adjustments = json!({ "contrast": 42 });
        let outcome = merge_style_proposal(
            &mut adjustments,
            &sample_proposal(0.5, -30.0),
            "m1",
            "t1",
            StyleMergePolicy::Overwrite,
        );
        assert_eq!(outcome.manual, vec!["contrast".to_owned()]);
        assert_eq!(adjustments["contrast"], json!(20.0));
        assert_eq!(adjustments[STYLE_MARKER_KEY]["previous"]["contrast"], json!(42.0));
        assert_eq!(adjustments[STYLE_MARKER_KEY]["keys"].as_array().map(Vec::len), Some(10));
    }

    #[test]
    fn merge_into_missing_adjustments_creates_them() {
        let proposal = to_proposal(&[0.1; 10]);
        let mut adjustments = Value::Null;
        let outcome =
            merge_style_proposal(&mut adjustments, &proposal, "m", "t", StyleMergePolicy::SkipEdited);
        assert_eq!(outcome.applied.len(), 10);
        assert_eq!(adjustments["exposure"], json!(0.1));
    }

    #[test]
    fn lightroom_fixture_converts_like_the_python_trainer() {
        let xmp = fs::read_to_string(fixtures().join("lightroom-basic.xmp")).expect("xmp");
        let expected = read_json(&fixtures().join("lightroom-basic.expected.json"));
        let preset = crate::preset_converter::convert_xmp_to_preset(&xmp).expect("preset");
        for key in OUTPUT_KEYS {
            let actual = preset.adjustments[key].as_f64().expect(key);
            let wanted = expected[key].as_f64().expect(key);
            assert!((actual - wanted).abs() < 1e-6, "{key}: {actual} != {wanted}");
        }
    }

    fn style_batch_merge(
        adjustments: &mut Value,
        proposal: &Value,
    ) -> (StyleMergeOutcome, bool) {
        let proposal = proposal.as_object().cloned().unwrap_or_default();
        let outcome =
            merge_style_proposal(adjustments, &proposal, "m1", "t1", StyleMergePolicy::SkipEdited);
        let write = !outcome.skipped;
        (outcome, write)
    }

    const MANUAL_RATING_XMP: &str = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="5"/>
 </rdf:RDF>
</x:xmpmeta>
"#;

    #[test]
    fn skipped_photo_leaves_sidecar_and_xmp_bytes_untouched() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let image = directory.path().join("photo.jpg");
        let sidecar = directory.path().join("photo.jpg.rrdata");
        let xmp = directory.path().join("photo.xmp");
        let sidecar_bytes = br#"{"version":1,"rating":0,"adjustments":{"contrast":42}}"#;
        fs::write(&sidecar, sidecar_bytes).expect("write sidecar");
        fs::write(&xmp, MANUAL_RATING_XMP).expect("write XMP");

        let (outcome, written) = crate::file_management::update_sidecar_adjustments(
            &image,
            &sidecar,
            &Value::Object(sample_proposal(0.5, -30.0)),
            style_batch_merge,
            true,
            true,
        )
        .expect("skipped, not failed");

        assert!(outcome.skipped);
        assert!(!written);
        assert_eq!(fs::read(&sidecar).expect("sidecar"), sidecar_bytes);
        assert_eq!(fs::read_to_string(&xmp).expect("XMP"), MANUAL_RATING_XMP);
    }

    #[test]
    fn invalid_sidecar_is_refused_not_replaced() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let image = directory.path().join("photo.jpg");
        let sidecar = directory.path().join("photo.jpg.rrdata");
        let xmp = directory.path().join("photo.xmp");
        let sidecar_bytes = b"{\"version\":1,\"rating\":4,\"adjustments\":{\"contrast\":";
        fs::write(&sidecar, sidecar_bytes).expect("write truncated sidecar");
        fs::write(&xmp, MANUAL_RATING_XMP).expect("write XMP");

        let error = crate::file_management::update_sidecar_adjustments(
            &image,
            &sidecar,
            &Value::Object(sample_proposal(0.5, -30.0)),
            style_batch_merge,
            true,
            true,
        )
        .err()
        .expect("invalid sidecar must fail");

        assert!(error.starts_with("SIDECAR_INVALID"), "{error}");
        assert_eq!(fs::read(&sidecar).expect("sidecar"), sidecar_bytes);
        assert_eq!(fs::read_to_string(&xmp).expect("XMP"), MANUAL_RATING_XMP);
    }

    #[test]
    fn styled_photo_keeps_the_xmp_rating() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let image = directory.path().join("photo.jpg");
        let sidecar = directory.path().join("photo.jpg.rrdata");
        let xmp = directory.path().join("photo.xmp");
        fs::write(&sidecar, br#"{"version":1,"rating":0,"adjustments":{"clarity":7}}"#)
            .expect("write sidecar");
        fs::write(&xmp, MANUAL_RATING_XMP).expect("write XMP");

        let (outcome, written) = crate::file_management::update_sidecar_adjustments(
            &image,
            &sidecar,
            &Value::Object(sample_proposal(0.5, -30.0)),
            style_batch_merge,
            true,
            true,
        )
        .expect("style applied");

        assert!(!outcome.skipped);
        assert!(written);
        let saved: Value = serde_json::from_slice(&fs::read(&sidecar).expect("sidecar")).expect("JSON");
        assert_eq!(saved["rating"], json!(5));
        assert_eq!(saved["adjustments"]["clarity"], json!(7));
        assert_eq!(saved["adjustments"]["exposure"], json!(0.5));
        assert!(saved["adjustments"][STYLE_MARKER_KEY].is_object());
        let xmp_content = fs::read_to_string(&xmp).expect("XMP");
        assert!(xmp_content.contains(r#"xmp:Rating="5""#), "{xmp_content}");
    }
}
