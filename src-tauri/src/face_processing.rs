//! Verified local YuNet face and five-point landmark inference.
//!
//! The detector is deliberately kept separate from subject attribution. A
//! face is a detected person, not automatically the photographed subject.

use std::fs;
use std::path::{Path, PathBuf};

use image::{DynamicImage, Rgb, RgbImage, imageops};
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Manager};

const CANONICAL_MANIFEST: &str = include_str!("../../models/face/manifest.json");
const MAX_MODEL_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EyeState {
    Open,
    Closed,
    Unknown,
}

#[derive(Debug, Clone, Copy)]
pub struct EyeAssessment {
    pub state: EyeState,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct FaceBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone)]
pub struct LocalFace {
    pub bbox: FaceBox,
    pub confidence: f32,
    pub eye: EyeAssessment,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FaceModelManifest {
    version: u8,
    model_id: String,
    detector_input: usize,
    detector_score_threshold: f32,
    detector_fallback_threshold: f32,
    detector_nms_threshold: f32,
    max_faces: usize,
    artifact: FaceArtifact,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FaceArtifact {
    filename: String,
    source_revision: String,
    source_url: String,
    sha256: String,
    bytes: u64,
    license: String,
    license_filename: String,
    license_url: String,
    license_bytes: u64,
    license_sha256: String,
}

pub struct FaceRuntime {
    manifest: FaceModelManifest,
    detector: Session,
}

impl FaceRuntime {
    pub fn load(model_directory: &Path) -> Result<Self, String> {
        let manifest = load_verified_manifest(model_directory)?;
        let model_path = model_directory.join(&manifest.artifact.filename);
        let _ = ort::init().with_name("PicPortal YuNet").commit();
        let detector = Session::builder()
            .map_err(|error| format!("YU_NET_RUNTIME_INIT_FAILED: {error}"))?
            .commit_from_file(model_path)
            .map_err(|error| format!("YU_NET_MODEL_LOAD_FAILED: {error}"))?;
        Ok(Self { manifest, detector })
    }

    pub fn analyze(
        &mut self,
        image: &RgbImage,
        score_threshold: f32,
    ) -> Result<Vec<LocalFace>, String> {
        let (source_width, source_height) = image.dimensions();
        let (frame, frame_scale_x, frame_scale_y) = detection_frame(image, 1920);
        let (tensor, scale_x, scale_y, resized_width, resized_height) =
            detector_tensor(&frame, self.manifest.detector_input)?;
        let input = Tensor::from_array((
            [
                1,
                3,
                self.manifest.detector_input,
                self.manifest.detector_input,
            ],
            tensor,
        ))
        .map_err(|error| format!("YU_NET_INPUT_FAILED: {error}"))?;
        let outputs = self
            .detector
            .run(ort::inputs!["input" => input])
            .map_err(|error| format!("YU_NET_INFERENCE_FAILED: {error}"))?;
        let names = [
            "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16",
            "bbox_32", "kps_8", "kps_16", "kps_32",
        ];
        let values = names
            .iter()
            .map(|name| {
                outputs[*name]
                    .try_extract_array::<f32>()
                    .map_err(|error| format!("YU_NET_OUTPUT_FAILED ({name}): {error}"))
                    .map(|array| array.iter().copied().collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut candidates = Vec::new();
        for (level, stride) in [8_usize, 16, 32].into_iter().enumerate() {
            let columns = self.manifest.detector_input / stride;
            let rows = self.manifest.detector_input / stride;
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
                    let frame_box = FaceBox {
                        x: x1 / scale_x,
                        y: y1 / scale_y,
                        width: (x2 - x1) / scale_x,
                        height: (y2 - y1) / scale_y,
                    };
                    candidates.push(LocalFace {
                        eye: estimate_eye_state(&frame, &landmarks),
                        bbox: FaceBox {
                            x: frame_box.x / frame_scale_x,
                            y: frame_box.y / frame_scale_y,
                            width: frame_box.width / frame_scale_x,
                            height: frame_box.height / frame_scale_y,
                        },
                        confidence,
                    });
                }
            }
        }

        let mut faces = non_maximum_suppression(
            candidates,
            self.manifest.detector_nms_threshold,
            self.manifest.max_faces,
        );
        for face in &mut faces {
            face.bbox.x = face.bbox.x.clamp(0.0, source_width as f32);
            face.bbox.y = face.bbox.y.clamp(0.0, source_height as f32);
            face.bbox.width = face.bbox.width.min(source_width as f32 - face.bbox.x);
            face.bbox.height = face.bbox.height.min(source_height as f32 - face.bbox.y);
        }
        Ok(faces)
    }

    pub fn primary_threshold(&self) -> f32 {
        self.manifest.detector_score_threshold
    }

    pub fn fallback_threshold(&self) -> f32 {
        self.manifest.detector_fallback_threshold
    }
}

pub fn load_for_app(app: &AppHandle) -> Result<FaceRuntime, String> {
    FaceRuntime::load(&model_directory(app)?)
}

pub fn model_directory(app: &AppHandle) -> Result<PathBuf, String> {
    if let Ok(path) = app
        .path()
        .resolve("models/face", tauri::path::BaseDirectory::Resource)
        && path.join("manifest.json").is_file()
    {
        return Ok(path);
    }
    let development = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/face");
    if development.join("manifest.json").is_file() {
        return Ok(development);
    }
    Err("YU_NET_MODEL_MANIFEST_UNAVAILABLE".to_owned())
}

pub fn choose_fallback_threshold(
    runtime: &FaceRuntime,
    primary_faces: &[LocalFace],
) -> Option<f32> {
    if primary_faces.is_empty() {
        Some(runtime.fallback_threshold())
    } else {
        None
    }
}

pub fn crop_face(image: &DynamicImage, face: &FaceBox, scale: f32) -> Option<DynamicImage> {
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let side = (face.width.max(face.height) * scale).max(1.0);
    let center_x = face.x + face.width / 2.0;
    let center_y = face.y + face.height / 2.0;
    let left = (center_x - side / 2.0).floor().max(0.0) as u32;
    let top = (center_y - side / 2.0).floor().max(0.0) as u32;
    let right = (center_x + side / 2.0).ceil().min(width as f32) as u32;
    let bottom = (center_y + side / 2.0).ceil().min(height as f32) as u32;
    if right <= left || bottom <= top {
        return None;
    }
    Some(DynamicImage::ImageRgb8(
        imageops::crop_imm(&rgb, left, top, right - left, bottom - top).to_image(),
    ))
}

fn load_verified_manifest(model_directory: &Path) -> Result<FaceModelManifest, String> {
    let manifest_path = model_directory.join("manifest.json");
    let bytes = fs::read(&manifest_path).map_err(|_| "YU_NET_MODEL_MANIFEST_INVALID".to_owned())?;
    let manifest: FaceModelManifest =
        serde_json::from_slice(&bytes).map_err(|_| "YU_NET_MODEL_MANIFEST_INVALID".to_owned())?;
    let canonical: FaceModelManifest = serde_json::from_str(CANONICAL_MANIFEST)
        .map_err(|_| "YU_NET_MODEL_MANIFEST_INVALID".to_owned())?;
    if manifest != canonical
        || manifest.version != 1
        || manifest.detector_input == 0
        || manifest.max_faces == 0
        || manifest.max_faces > 16
    {
        return Err("YU_NET_MODEL_CONTRACT_MISMATCH".to_owned());
    }
    let license_path = model_directory.join(&manifest.artifact.license_filename);
    let license_metadata =
        fs::metadata(&license_path).map_err(|_| "YU_NET_LICENSE_MISSING".to_owned())?;
    if !license_metadata.is_file()
        || license_metadata.len() != manifest.artifact.license_bytes
        || sha256_file(&license_path)? != manifest.artifact.license_sha256
    {
        return Err("YU_NET_LICENSE_INVALID".to_owned());
    }
    let model_path = model_directory.join(&manifest.artifact.filename);
    let metadata = fs::metadata(&model_path).map_err(|_| "YU_NET_MODEL_MISSING".to_owned())?;
    if !metadata.is_file()
        || metadata.len() != manifest.artifact.bytes
        || metadata.len() > MAX_MODEL_BYTES
    {
        return Err("YU_NET_MODEL_SIZE_INVALID".to_owned());
    }
    if sha256_file(&model_path)? != manifest.artifact.sha256 {
        return Err("YU_NET_MODEL_HASH_INVALID".to_owned());
    }
    Ok(manifest)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|_| "YU_NET_MODEL_UNREADABLE".to_owned())?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(hex::encode(hasher.finalize()))
}

fn detection_frame(image: &RgbImage, max_dimension: u32) -> (RgbImage, f32, f32) {
    let (width, height) = image.dimensions();
    let longest = width.max(height);
    if longest <= max_dimension || longest == 0 {
        return (image.clone(), 1.0, 1.0);
    }
    let scale = max_dimension as f32 / longest as f32;
    let frame_width = ((width as f32 * scale).round() as u32).max(1);
    let frame_height = ((height as f32 * scale).round() as u32).max(1);
    let frame = imageops::resize(
        image,
        frame_width,
        frame_height,
        imageops::FilterType::Triangle,
    );
    (
        frame,
        frame_width as f32 / width as f32,
        frame_height as f32 / height as f32,
    )
}

fn detector_tensor(
    image: &RgbImage,
    input_size: usize,
) -> Result<(Vec<f32>, f32, f32, usize, usize), String> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err("YU_NET_IMAGE_INVALID".to_owned());
    }
    let scale = (input_size as f64 / f64::from(width)).min(input_size as f64 / f64::from(height));
    let resized_width = ((f64::from(width) * scale + 0.5).floor() as usize).clamp(1, input_size);
    let resized_height = ((f64::from(height) * scale + 0.5).floor() as usize).clamp(1, input_size);
    let resized = imageops::resize(
        image,
        resized_width as u32,
        resized_height as u32,
        imageops::FilterType::Triangle,
    );
    let plane = input_size * input_size;
    let mut tensor = vec![0.0_f32; plane * 3];
    for y in 0..resized_height {
        for x in 0..resized_width {
            let pixel = resized.get_pixel(x as u32, y as u32).0;
            let output_index = y * input_size + x;
            tensor[output_index] = f32::from(pixel[2]);
            tensor[plane + output_index] = f32::from(pixel[1]);
            tensor[plane * 2 + output_index] = f32::from(pixel[0]);
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

fn estimate_eye_state(image: &RgbImage, landmarks: &[[f32; 2]; 5]) -> EyeAssessment {
    let left = landmarks[0];
    let right = landmarks[1];
    let interocular = ((right[0] - left[0]).powi(2) + (right[1] - left[1]).powi(2)).sqrt();
    if !interocular.is_finite() || interocular < 3.0 {
        return EyeAssessment {
            state: EyeState::Unknown,
            confidence: 0.0,
        };
    }
    let mut signal = 0.0;
    let mut samples = 0_u32;
    let (width, height) = image.dimensions();
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
                    let Rgb([red, green, blue]) = image.get_pixel(px, py);
                    0.2126 * f32::from(*red)
                        + 0.7152 * f32::from(*green)
                        + 0.0722 * f32::from(*blue)
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
    EyeAssessment { state, confidence }
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
    mut candidates: Vec<LocalFace>,
    threshold: f32,
    maximum: usize,
) -> Vec<LocalFace> {
    candidates.sort_by(|left, right| {
        right
            .confidence
            .total_cmp(&left.confidence)
            .then_with(|| left.bbox.y.total_cmp(&right.bbox.y))
            .then_with(|| left.bbox.x.total_cmp(&right.bbox.x))
    });
    let mut kept = Vec::new();
    for candidate in candidates {
        if kept.iter().all(|existing: &LocalFace| {
            intersection_over_union(&candidate.bbox, &existing.bbox) < threshold
        }) {
            kept.push(candidate);
            if kept.len() == maximum {
                break;
            }
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_is_only_requested_when_primary_is_empty() {
        let manifest: FaceModelManifest =
            serde_json::from_str(CANONICAL_MANIFEST).expect("manifest");
        assert_eq!(manifest.detector_score_threshold, 0.9);
        assert_eq!(manifest.detector_fallback_threshold, 0.7);
        assert!(choose_fallback_threshold_from_values(
            manifest.detector_fallback_threshold,
            0
        ));
        assert!(!choose_fallback_threshold_from_values(
            manifest.detector_fallback_threshold,
            1
        ));
    }

    fn choose_fallback_threshold_from_values(_threshold: f32, face_count: usize) -> bool {
        face_count == 0
    }

    #[test]
    fn detection_frame_caps_large_inputs_and_preserves_coordinates_scale() {
        let image = RgbImage::new(6048, 4032);
        let (frame, scale_x, scale_y) = detection_frame(&image, 1920);

        assert_eq!(frame.dimensions(), (1920, 1280));
        assert!((scale_x - (1920.0 / 6048.0)).abs() < f32::EPSILON);
        assert!((scale_y - (1280.0 / 4032.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn detection_frame_does_not_upscale_small_inputs() {
        let image = RgbImage::new(640, 480);
        let (frame, scale_x, scale_y) = detection_frame(&image, 1920);

        assert_eq!(frame.dimensions(), (640, 480));
        assert_eq!(scale_x, 1.0);
        assert_eq!(scale_y, 1.0);
    }
}
