//! Cross-platform local YuNet/SFace processing for PicPortal.
//!
//! The same ONNX contract is used on macOS, Linux and Windows.  No server
//! inference fallback is hidden here: callers receive an explicit error when
//! the verified local bundle is unavailable.

use std::{
    fs,
    path::{Path, PathBuf},
};

use image::DynamicImage;
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use tauri::{AppHandle, Manager};

use crate::local_derivatives;

const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
const MAX_MODEL_BYTES: u64 = 128 * 1024 * 1024;
const MAX_EMBEDDING_DIMENSION: usize = 1024;
const MAX_DETECTOR_INPUT: usize = 1280;
const MAX_FACES: usize = 16;
const CANONICAL_MANIFEST: &str = include_str!("../../models/face/manifest.json");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EyeState {
    Open,
    Closed,
    Unknown,
    NotApplicable,
}

#[derive(Debug, Clone, Copy)]
pub struct EyeAssessment {
    pub state: EyeState,
    pub confidence: f32,
    /// The culling signal is explicitly heuristic; it is not a learned eye model.
    pub method: &'static str,
}

#[derive(Debug, Clone)]
pub struct FaceBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug)]
pub struct LocalFace {
    pub bbox: FaceBox,
    pub confidence: f32,
    pub embedding: Vec<f32>,
    pub thumbnail: Vec<u8>,
    pub thumbnail_sha256: String,
    pub eye: EyeAssessment,
}

#[derive(Debug)]
pub struct CullingFace {
    pub bbox: FaceBox,
    pub confidence: f32,
    pub eye: EyeAssessment,
}

#[derive(Debug)]
pub struct LocalFaceAnalysis {
    pub source_sha256: String,
    pub source_width: u32,
    pub source_height: u32,
    pub faces: Vec<LocalFace>,
    pub model_id: String,
    pub model_digest: String,
    pub pipeline_version: String,
    pub embedding_dimension: usize,
    pub detector_input: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FaceModelContract {
    pub version: u8,
    pub model_id: String,
    pub bundle_sha256: String,
    pub pipeline_version: String,
    pub embedding_dimension: usize,
    pub detector_input: usize,
    pub recognizer_input: usize,
    pub detector_score_threshold: f32,
    pub detector_nms_threshold: f32,
    pub cosine_distance_threshold: f32,
    pub max_faces: usize,
    pub face_thumbnail: FaceThumbnailContract,
    pub runtime_compatibility: RuntimeCompatibility,
    pub artifacts: Vec<ModelArtifact>,
    pub test_fixture: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FaceThumbnailContract {
    pub width: u32,
    pub height: u32,
    pub quality: f32,
    pub crop_scale: f32,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RuntimeCompatibility {
    pub rust_crate_version: String,
    pub rust_runtime_version: String,
    pub python_package_version: String,
    pub embedding_cosine_minimum: f32,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelArtifact {
    pub role: String,
    pub filename: String,
    pub source_revision: String,
    pub source_url: String,
    pub sha256: String,
    pub bytes: u64,
    pub license: String,
    pub license_filename: String,
    pub license_url: String,
    pub license_sha256: String,
}

impl FaceModelContract {
    pub fn artifact(&self, role: &str) -> Result<&ModelArtifact, String> {
        self.artifacts
            .iter()
            .find(|artifact| artifact.role == role)
            .ok_or_else(|| "FACE_MODEL_MANIFEST_INVALID".to_owned())
    }
}

pub struct FaceRuntime {
    pub contract: FaceModelContract,
    detector: Session,
    recognizer: Session,
}

impl FaceRuntime {
    pub fn load(model_directory: &Path) -> Result<Self, String> {
        let contract = load_verified_contract(model_directory)?;
        let detector_path = model_directory.join(&contract.artifact("detector")?.filename);
        let recognizer_path = model_directory.join(&contract.artifact("recognizer")?.filename);
        let _ = ort::init().with_name("PicPortal Face").commit();
        let detector = Session::builder()
            .map_err(|error| format!("FACE_RUNTIME_INIT_FAILED: {error}"))?
            .commit_from_file(detector_path)
            .map_err(|error| format!("FACE_DETECTOR_LOAD_FAILED: {error}"))?;
        let recognizer = Session::builder()
            .map_err(|error| format!("FACE_RUNTIME_INIT_FAILED: {error}"))?
            .commit_from_file(recognizer_path)
            .map_err(|error| format!("FACE_RECOGNIZER_LOAD_FAILED: {error}"))?;
        Ok(Self {
            contract,
            detector,
            recognizer,
        })
    }

    pub fn analyze(&mut self, source: &[u8]) -> Result<LocalFaceAnalysis, String> {
        let decoded = local_derivatives::decode_source(source)?;
        let rgb = decoded.image.to_rgb8();
        let (width, height) = rgb.dimensions();
        let candidates = self.detect(rgb.as_raw(), width, height)?;
        let mut faces = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let embedding = self.embed(rgb.as_raw(), width, height, &candidate.landmarks)?;
            let thumbnail = make_thumbnail(
                rgb.as_raw(),
                width,
                height,
                &candidate.bbox,
                &self.contract.face_thumbnail,
            )?;
            faces.push(LocalFace {
                bbox: normalize_bbox(&candidate.bbox, width, height)?,
                confidence: candidate.confidence,
                thumbnail_sha256: local_derivatives::sha256_hex(&thumbnail),
                thumbnail,
                eye: candidate.eye,
                embedding,
            });
        }
        Ok(LocalFaceAnalysis {
            source_sha256: decoded.source_sha256,
            source_width: decoded.width,
            source_height: decoded.height,
            model_id: self.contract.model_id.clone(),
            model_digest: self.contract.bundle_sha256.clone(),
            pipeline_version: self.contract.pipeline_version.clone(),
            embedding_dimension: self.contract.embedding_dimension,
            detector_input: self.contract.detector_input,
            faces,
        })
    }

    /// Runs YuNet detection/eye review only; culling never computes an SFace embedding.
    pub fn analyze_for_culling(
        &mut self,
        image: &DynamicImage,
        score_threshold: f32,
    ) -> Result<Vec<CullingFace>, String> {
        if !score_threshold.is_finite() || !(0.0..=1.0).contains(&score_threshold) {
            return Err("FACE_CULLING_THRESHOLD_INVALID".to_owned());
        }
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        self.detect_with_threshold(rgb.as_raw(), width, height, score_threshold)
            .map(|candidates| {
                candidates
                    .into_iter()
                    .map(|candidate| CullingFace {
                        bbox: candidate.bbox,
                        confidence: candidate.confidence,
                        eye: candidate.eye,
                    })
                    .collect()
            })
    }

    fn detect(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<Candidate>, String> {
        let score_threshold = self.contract.detector_score_threshold;
        self.detect_with_threshold(rgb, width, height, score_threshold)
    }

    fn detect_with_threshold(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
        score_threshold: f32,
    ) -> Result<Vec<Candidate>, String> {
        let input_size = self.contract.detector_input;
        let (input, scale_x, scale_y, resized_width, resized_height) =
            detector_tensor(rgb, width, height, input_size)?;
        let tensor = Tensor::from_array(([1, 3, input_size, input_size], input))
            .map_err(|error| format!("FACE_DETECTOR_INPUT_FAILED: {error}"))?;
        let outputs = self
            .detector
            .run(ort::inputs!["input" => tensor])
            .map_err(|error| format!("FACE_DETECTOR_INFERENCE_FAILED: {error}"))?;
        let names = [
            "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16",
            "bbox_32", "kps_8", "kps_16", "kps_32",
        ];
        let mut values = Vec::with_capacity(names.len());
        for name in names {
            let data = outputs[name]
                .try_extract_array::<f32>()
                .map_err(|error| format!("FACE_DETECTOR_OUTPUT_FAILED: {error}"))?
                .iter()
                .copied()
                .collect::<Vec<_>>();
            values.push(data);
        }
        let mut candidates = Vec::new();
        for (level, stride) in [8_usize, 16, 32].into_iter().enumerate() {
            let columns = input_size / stride;
            let rows = input_size / stride;
            for row in 0..rows {
                for column in 0..columns {
                    let index = row * columns + column;
                    let confidence = (values[level][index].clamp(0.0, 1.0)
                        * values[level + 3][index].clamp(0.0, 1.0))
                    .sqrt();
                    if confidence < score_threshold {
                        continue;
                    }
                    let boxes = &values[level + 6];
                    let points = &values[level + 9];
                    let center_x = (column as f32 + boxes[index * 4]) * stride as f32;
                    let center_y = (row as f32 + boxes[index * 4 + 1]) * stride as f32;
                    let box_width = boxes[index * 4 + 2].exp() * stride as f32;
                    let box_height = boxes[index * 4 + 3].exp() * stride as f32;
                    let x1 = (center_x - box_width / 2.0).max(0.0);
                    let y1 = (center_y - box_height / 2.0).max(0.0);
                    let x2 = (center_x + box_width / 2.0).min(resized_width as f32);
                    let y2 = (center_y + box_height / 2.0).min(resized_height as f32);
                    if x2 <= x1 || y2 <= y1 {
                        continue;
                    }
                    let mut landmarks = [[0.0; 2]; 5];
                    for point in 0..5 {
                        landmarks[point][0] = ((points[index * 10 + point * 2] + column as f32)
                            * stride as f32)
                            / scale_x;
                        landmarks[point][1] = ((points[index * 10 + point * 2 + 1] + row as f32)
                            * stride as f32)
                            / scale_y;
                    }
                    let bbox = FaceBox {
                        x: x1 / scale_x,
                        y: y1 / scale_y,
                        width: (x2 - x1) / scale_x,
                        height: (y2 - y1) / scale_y,
                    };
                    candidates.push(Candidate {
                        eye: estimate_eye_state(rgb, width, height, &landmarks),
                        bbox,
                        confidence,
                        landmarks,
                    });
                }
            }
        }
        Ok(non_maximum_suppression(
            candidates,
            self.contract.detector_nms_threshold,
            self.contract.max_faces,
        ))
    }

    fn embed(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
        landmarks: &[[f32; 2]; 5],
    ) -> Result<Vec<f32>, String> {
        let input_size = self.contract.recognizer_input;
        let input = aligned_face_tensor(rgb, width, height, landmarks, input_size)?;
        let tensor = Tensor::from_array(([1, 3, input_size, input_size], input))
            .map_err(|error| format!("FACE_RECOGNIZER_INPUT_FAILED: {error}"))?;
        let outputs = self
            .recognizer
            .run(ort::inputs!["data" => tensor])
            .map_err(|error| format!("FACE_RECOGNIZER_INFERENCE_FAILED: {error}"))?;
        let values = outputs["fc1"]
            .try_extract_array::<f32>()
            .map_err(|error| format!("FACE_RECOGNIZER_OUTPUT_FAILED: {error}"))?
            .iter()
            .copied()
            .collect::<Vec<_>>();
        if values.len() != self.contract.embedding_dimension
            || !values.iter().all(|v| v.is_finite())
        {
            return Err("FACE_EMBEDDING_DIMENSION_INVALID".to_owned());
        }
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm <= f32::EPSILON {
            return Err("FACE_EMBEDDING_NORM_INVALID".to_owned());
        }
        Ok(values.iter().map(|value| value / norm).collect())
    }
}

#[derive(Clone)]
struct Candidate {
    bbox: FaceBox,
    confidence: f32,
    landmarks: [[f32; 2]; 5],
    eye: EyeAssessment,
}

pub fn model_directory(app: &AppHandle) -> Result<PathBuf, String> {
    if let Ok(path) = app
        .path()
        .resolve("models/face", tauri::path::BaseDirectory::Resource)
    {
        if path.join("manifest.json").is_file() {
            return Ok(path);
        }
    }
    let development = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/face");
    if development.join("manifest.json").is_file() {
        return Ok(development);
    }
    Err("FACE_MODEL_BUNDLE_UNAVAILABLE".to_owned())
}

pub fn load_verified_contract(model_directory: &Path) -> Result<FaceModelContract, String> {
    let manifest_path = model_directory.join("manifest.json");
    let metadata =
        fs::metadata(&manifest_path).map_err(|_| "FACE_MODEL_MANIFEST_INVALID".to_owned())?;
    if !metadata.is_file() || metadata.len() > MAX_MANIFEST_BYTES {
        return Err("FACE_MODEL_MANIFEST_INVALID".to_owned());
    }
    let bytes = fs::read(&manifest_path).map_err(|_| "FACE_MODEL_MANIFEST_INVALID".to_owned())?;
    let contract: FaceModelContract =
        serde_json::from_slice(&bytes).map_err(|_| "FACE_MODEL_MANIFEST_INVALID".to_owned())?;
    let canonical: FaceModelContract = serde_json::from_str(CANONICAL_MANIFEST)
        .map_err(|_| "FACE_MODEL_MANIFEST_INVALID".to_owned())?;
    if contract != canonical {
        return Err("FACE_MODEL_CONTRACT_MISMATCH".to_owned());
    }
    if contract.version != 1
        || contract.embedding_dimension == 0
        || contract.embedding_dimension > MAX_EMBEDDING_DIMENSION
        || !(160..=MAX_DETECTOR_INPUT).contains(&contract.detector_input)
        || !contract.detector_input.is_multiple_of(32)
        || contract.recognizer_input != 112
        || contract.max_faces == 0
        || contract.max_faces > MAX_FACES
        || contract.face_thumbnail.width != 220
        || contract.face_thumbnail.height != 220
        || !contract.detector_score_threshold.is_finite()
        || !contract.detector_nms_threshold.is_finite()
    {
        return Err("FACE_MODEL_MANIFEST_INVALID".to_owned());
    }
    if contract.artifacts.len() != 2 {
        return Err("FACE_MODEL_MANIFEST_INVALID".to_owned());
    }
    let mut descriptor = String::new();
    for artifact in &contract.artifacts {
        let path = model_directory.join(&artifact.filename);
        let metadata = fs::metadata(&path).map_err(|_| "FACE_MODEL_ARTIFACT_MISSING".to_owned())?;
        if !metadata.is_file()
            || metadata.len() != artifact.bytes
            || metadata.len() > MAX_MODEL_BYTES
        {
            return Err("FACE_MODEL_ARTIFACT_SIZE_INVALID".to_owned());
        }
        let content = fs::read(&path).map_err(|_| "FACE_MODEL_ARTIFACT_UNREADABLE".to_owned())?;
        if local_derivatives::sha256_hex(&content) != artifact.sha256 {
            return Err(format!(
                "FACE_MODEL_ARTIFACT_HASH_INVALID: {}",
                artifact.filename
            ));
        }
        descriptor.push_str(&format!("{} {}\n", artifact.filename, artifact.sha256));
    }
    if local_derivatives::sha256_hex(descriptor.as_bytes()) != contract.bundle_sha256 {
        return Err("FACE_MODEL_BUNDLE_DIGEST_MISMATCH".to_owned());
    }
    Ok(contract)
}

pub fn ensure_runtime<'a>(
    runtime: &'a mut Option<FaceRuntime>,
    app: &AppHandle,
) -> Result<&'a mut FaceRuntime, String> {
    if runtime.is_none() {
        let directory = model_directory(app)?;
        *runtime = Some(FaceRuntime::load(&directory)?);
    }
    runtime
        .as_mut()
        .ok_or_else(|| "FACE_RUNTIME_UNAVAILABLE".to_owned())
}

fn detector_tensor(
    rgb: &[u8],
    width: u32,
    height: u32,
    input_size: usize,
) -> Result<(Vec<f32>, f32, f32, usize, usize), String> {
    if width == 0 || height == 0 || rgb.len() != (u64::from(width) * u64::from(height) * 3) as usize
    {
        return Err("FACE_IMAGE_INVALID".to_owned());
    }
    let scale = (input_size as f64 / f64::from(width)).min(input_size as f64 / f64::from(height));
    let resized_width = ((f64::from(width) * scale + 0.5).floor() as usize).clamp(1, input_size);
    let resized_height = ((f64::from(height) * scale + 0.5).floor() as usize).clamp(1, input_size);
    let resized = resize_bilinear(
        rgb,
        width,
        height,
        resized_width as u32,
        resized_height as u32,
    )?;
    let plane = input_size * input_size;
    let mut tensor = vec![0.0_f32; plane * 3];
    for y in 0..resized_height {
        for x in 0..resized_width {
            let input_index = (y * resized_width + x) * 3;
            let output_index = y * input_size + x;
            tensor[output_index] = f32::from(resized[input_index + 2]);
            tensor[plane + output_index] = f32::from(resized[input_index + 1]);
            tensor[plane * 2 + output_index] = f32::from(resized[input_index]);
        }
    }
    Ok((
        tensor,
        resized_width as f32 / width as f32,
        resized_height as f32 / height as f32,
        resized_width,
        resized_height,
    ))
}

fn estimate_eye_state(
    rgb: &[u8],
    width: u32,
    height: u32,
    landmarks: &[[f32; 2]; 5],
) -> EyeAssessment {
    let left = landmarks[0];
    let right = landmarks[1];
    let interocular = ((right[0] - left[0]).powi(2) + (right[1] - left[1]).powi(2)).sqrt();
    if !interocular.is_finite() || interocular < 3.0 {
        return EyeAssessment {
            state: EyeState::Unknown,
            confidence: 0.0,
            method: "local-heuristic",
        };
    }
    let mut signal = 0.0;
    let mut samples = 0_u32;
    for [cx, cy] in [left, right] {
        let radius_x = (interocular * 0.28).max(2.0);
        let radius_y = (interocular * 0.16).max(2.0);
        let x0 = (cx - radius_x).floor().max(1.0) as u32;
        let x1 = (cx + radius_x).ceil().min(width.saturating_sub(2) as f32) as u32;
        let y0 = (cy - radius_y).floor().max(1.0) as u32;
        let y1 = (cy + radius_y).ceil().min(height.saturating_sub(2) as f32) as u32;
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        for y in y0..y1 {
            for x in x0..x1 {
                let luma = |px: u32, py: u32| {
                    let index = ((py * width + px) * 3) as usize;
                    0.2126 * f32::from(rgb[index])
                        + 0.7152 * f32::from(rgb[index + 1])
                        + 0.0722 * f32::from(rgb[index + 2])
                };
                signal += (luma(x + 1, y) - luma(x - 1, y)).abs();
                samples += 1;
            }
        }
    }
    if samples == 0 {
        return EyeAssessment {
            state: EyeState::Unknown,
            confidence: 0.0,
            method: "local-heuristic",
        };
    }
    let score = signal / samples as f32;
    let (state, confidence) = if score >= 18.0 {
        (EyeState::Open, ((score - 18.0) / 18.0).clamp(0.0, 1.0))
    } else if score <= 7.0 {
        (EyeState::Closed, ((7.0 - score) / 7.0).clamp(0.0, 1.0))
    } else {
        (EyeState::Unknown, 0.0)
    };
    EyeAssessment {
        state,
        confidence,
        method: "local-heuristic",
    }
}

fn resize_bilinear(
    rgb: &[u8],
    width: u32,
    height: u32,
    output_width: u32,
    output_height: u32,
) -> Result<Vec<u8>, String> {
    if output_width == 0 || output_height == 0 {
        return Err("FACE_RESIZE_INVALID".to_owned());
    }
    let mut output = vec![0_u8; (u64::from(output_width) * u64::from(output_height) * 3) as usize];
    for output_y in 0..output_height {
        let source_y = ((f64::from(output_y) + 0.5) * f64::from(height) / f64::from(output_height)
            - 0.5)
            .clamp(0.0, f64::from(height - 1));
        for output_x in 0..output_width {
            let source_x =
                ((f64::from(output_x) + 0.5) * f64::from(width) / f64::from(output_width) - 0.5)
                    .clamp(0.0, f64::from(width - 1));
            let x0 = source_x.floor() as u32;
            let y0 = source_y.floor() as u32;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);
            let wx = (source_x - f64::from(x0)) as f32;
            let wy = (source_y - f64::from(y0)) as f32;
            let pixel = |px: u32, py: u32, channel: usize| {
                f32::from(rgb[((py * width + px) * 3) as usize + channel])
            };
            let index = ((output_y * output_width + output_x) * 3) as usize;
            for channel in 0..3 {
                let top = pixel(x0, y0, channel) * (1.0 - wx) + pixel(x1, y0, channel) * wx;
                let bottom = pixel(x0, y1, channel) * (1.0 - wx) + pixel(x1, y1, channel) * wx;
                output[index + channel] = (top * (1.0 - wy) + bottom * wy)
                    .round_ties_even()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(output)
}

fn aligned_face_tensor(
    rgb: &[u8],
    width: u32,
    height: u32,
    landmarks: &[[f32; 2]; 5],
    input_size: usize,
) -> Result<Vec<f32>, String> {
    const DESTINATION: [[f64; 2]; 5] = [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2049],
    ];
    let source_mean = landmarks.iter().fold([0.0_f64; 2], |mut mean, point| {
        mean[0] += f64::from(point[0]) / 5.0;
        mean[1] += f64::from(point[1]) / 5.0;
        mean
    });
    let destination_mean = DESTINATION.iter().fold([0.0_f64; 2], |mut mean, point| {
        mean[0] += point[0] / 5.0;
        mean[1] += point[1] / 5.0;
        mean
    });
    let mut denominator = 0.0;
    let mut a_numerator = 0.0;
    let mut b_numerator = 0.0;
    for (source, destination) in landmarks.iter().zip(DESTINATION) {
        let sx = f64::from(source[0]) - source_mean[0];
        let sy = f64::from(source[1]) - source_mean[1];
        let dx = destination[0] - destination_mean[0];
        let dy = destination[1] - destination_mean[1];
        denominator += sx * sx + sy * sy;
        a_numerator += sx * dx + sy * dy;
        b_numerator += sx * dy - sy * dx;
    }
    if !denominator.is_finite() || denominator <= 1e-9 {
        return Err("FACE_LANDMARKS_INVALID".to_owned());
    }
    let a = a_numerator / denominator;
    let b = b_numerator / denominator;
    let tx = destination_mean[0] - a * source_mean[0] + b * source_mean[1];
    let ty = destination_mean[1] - b * source_mean[0] - a * source_mean[1];
    let inverse_denominator = a * a + b * b;
    if inverse_denominator <= 1e-12 {
        return Err("FACE_LANDMARKS_INVALID".to_owned());
    }
    let plane = input_size * input_size;
    let mut tensor = vec![0.0_f32; plane * 3];
    for y in 0..input_size {
        for x in 0..input_size {
            let shifted_x = x as f64 - tx;
            let shifted_y = y as f64 - ty;
            let source_x = (a * shifted_x + b * shifted_y) / inverse_denominator;
            let source_y = (-b * shifted_x + a * shifted_y) / inverse_denominator;
            let sampled = sample_rgb(rgb, width, height, source_x, source_y);
            let index = y * input_size + x;
            tensor[index] = sampled[0];
            tensor[plane + index] = sampled[1];
            tensor[plane * 2 + index] = sampled[2];
        }
    }
    Ok(tensor)
}

fn sample_rgb(rgb: &[u8], width: u32, height: u32, x: f64, y: f64) -> [f32; 3] {
    if x < 0.0 || y < 0.0 || x > f64::from(width - 1) || y > f64::from(height - 1) {
        return [0.0; 3];
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let wx = (x - f64::from(x0)) as f32;
    let wy = (y - f64::from(y0)) as f32;
    let pixel = |px: u32, py: u32, channel: usize| {
        f32::from(rgb[((py * width + px) * 3) as usize + channel])
    };
    let mut output = [0.0; 3];
    for (channel, value) in output.iter_mut().enumerate() {
        let top = pixel(x0, y0, channel) * (1.0 - wx) + pixel(x1, y0, channel) * wx;
        let bottom = pixel(x0, y1, channel) * (1.0 - wx) + pixel(x1, y1, channel) * wx;
        *value = top * (1.0 - wy) + bottom * wy;
    }
    output
}

fn intersection_over_union(left: &FaceBox, right: &FaceBox) -> f32 {
    let width = ((left.x + left.width).min(right.x + right.width) - left.x.max(right.x)).max(0.0);
    let height =
        ((left.y + left.height).min(right.y + right.height) - left.y.max(right.y)).max(0.0);
    let intersection = width * height;
    let union = left.width * left.height + right.width * right.height - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

fn non_maximum_suppression(
    mut candidates: Vec<Candidate>,
    threshold: f32,
    maximum: usize,
) -> Vec<Candidate> {
    candidates.sort_by(|left, right| {
        right
            .confidence
            .total_cmp(&left.confidence)
            .then_with(|| left.bbox.y.total_cmp(&right.bbox.y))
            .then_with(|| left.bbox.x.total_cmp(&right.bbox.x))
    });
    let mut kept: Vec<Candidate> = Vec::new();
    for candidate in candidates {
        if kept
            .iter()
            .all(|existing| intersection_over_union(&candidate.bbox, &existing.bbox) < threshold)
        {
            kept.push(candidate);
            if kept.len() == maximum {
                break;
            }
        }
    }
    kept
}

fn normalize_bbox(bbox: &FaceBox, width: u32, height: u32) -> Result<FaceBox, String> {
    let width = width as f32;
    let height = height as f32;
    let x = (bbox.x / width).clamp(0.0, 1.0);
    let y = (bbox.y / height).clamp(0.0, 1.0);
    let right = ((bbox.x + bbox.width.max(0.0)) / width).clamp(x, 1.0);
    let bottom = ((bbox.y + bbox.height.max(0.0)) / height).clamp(y, 1.0);
    let normalized = FaceBox {
        x,
        y,
        width: right - x,
        height: bottom - y,
    };
    if normalized.width <= 0.0 || normalized.height <= 0.0 {
        return Err("FACE_BBOX_INVALID".to_owned());
    }
    Ok(normalized)
}

fn make_thumbnail(
    rgb: &[u8],
    width: u32,
    height: u32,
    bbox: &FaceBox,
    contract: &FaceThumbnailContract,
) -> Result<Vec<u8>, String> {
    let size = bbox.width.max(bbox.height).max(1.0) * contract.crop_scale;
    let center_x = bbox.x + bbox.width / 2.0;
    let center_y = bbox.y + bbox.height / 2.0;
    let left = (center_x - size / 2.0 + 0.5).floor().max(0.0) as u32;
    let top = (center_y - size / 2.0 + 0.5).floor().max(0.0) as u32;
    let right = ((center_x + size / 2.0 + 0.5).floor() as u32).min(width);
    let bottom = ((center_y + size / 2.0 + 0.5).floor() as u32).min(height);
    if right <= left || bottom <= top {
        return Err("INVALID_FACE_CROP".to_owned());
    }
    let crop_width = right - left;
    let crop_height = bottom - top;
    let mut crop = vec![0_u8; (u64::from(crop_width) * u64::from(crop_height) * 3) as usize];
    for y in 0..crop_height {
        let source_start = (((top + y) * width + left) * 3) as usize;
        let destination_start = (y * crop_width * 3) as usize;
        let bytes = (crop_width * 3) as usize;
        crop[destination_start..destination_start + bytes]
            .copy_from_slice(&rgb[source_start..source_start + bytes]);
    }
    let resized = resize_bilinear(
        &crop,
        crop_width,
        crop_height,
        contract.width,
        contract.height,
    )?;
    Ok(
        webp::Encoder::from_rgb(&resized, contract.width, contract.height)
            .encode(contract.quality)
            .to_vec(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_contract_is_compatible_with_runtime_limits() {
        let contract: FaceModelContract =
            serde_json::from_str(CANONICAL_MANIFEST).expect("manifest");
        assert_eq!(contract.model_id, "opencv-yunet-sface-v1");
        assert_eq!(contract.embedding_dimension, 128);
        assert_eq!(contract.detector_input, 640);
        assert_eq!(contract.recognizer_input, 112);
        assert_eq!(contract.max_faces, 16);
    }

    #[test]
    fn canonical_fixture_runs_local_yunet_sface() {
        let manifest_directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let model_directory = manifest_directory.join("../models/face");
        let ort_library_name = if cfg!(target_os = "windows") {
            "onnxruntime.dll"
        } else if cfg!(target_os = "macos") {
            "libonnxruntime.dylib"
        } else {
            "libonnxruntime.so"
        };
        let ort_library = manifest_directory.join(format!("resources/{ort_library_name}"));
        if ort_library.is_file() {
            unsafe { std::env::set_var("ORT_DYLIB_PATH", &ort_library) };
        }
        let source = std::fs::read(model_directory.join("fixtures/picportal-synthetic-face.jpg"))
            .expect("fixture");
        let expected: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_directory.join("fixtures/picportal-synthetic-face.expected.json"))
                .expect("expected fixture"),
        )
        .expect("expected JSON");
        let mut runtime = FaceRuntime::load(&model_directory).expect("runtime");
        let analysis = runtime.analyze(&source).expect("analysis");
        assert_eq!(analysis.faces.len(), 1);
        let expected_face = &expected["faces"][0];
        let expected_embedding = expected_face["embedding"].as_array().expect("embedding");
        let cosine = analysis.faces[0]
            .embedding
            .iter()
            .zip(expected_embedding)
            .map(|(actual, expected)| actual * expected.as_f64().expect("finite") as f32)
            .sum::<f32>();
        assert!(cosine > 0.999, "fixture embedding cosine was {cosine}");
        assert!(
            (analysis.faces[0].bbox.x - expected_face["bbox"]["x"].as_f64().unwrap() as f32).abs()
                < 0.002
        );
        assert!(analysis.faces[0].thumbnail.starts_with(b"RIFF"));
    }

    #[test]
    fn resize_uses_center_of_pixel_sampling() {
        let source = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let resized = resize_bilinear(&source, 2, 2, 3, 3).expect("resize");
        assert_eq!(&resized[12..15], &[128, 128, 128]);
    }

    #[test]
    fn missing_model_bundle_fails_closed() {
        let path = std::env::temp_dir().join(format!("picportal-face-{}", uuid::Uuid::new_v4()));
        assert!(load_verified_contract(&path).is_err());
    }
}
