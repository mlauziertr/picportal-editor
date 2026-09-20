use crate::app_settings::load_settings;
use image::{GenericImageView, GrayImage, imageops};
use image_hasher::{HashAlg, HasherConfig};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tauri::{AppHandle, Emitter, State};

use crate::{PicPortalState, face_processing, image_loader};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase", default)]
pub struct StarMapping {
    pub retained: u8,
    pub review: u8,
    pub duplicate: u8,
    pub defect: u8,
    pub unknown: u8,
}

impl Default for StarMapping {
    fn default() -> Self {
        Self {
            retained: 5,
            review: 3,
            duplicate: 1,
            defect: 0,
            unknown: 2,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase", default)]
pub struct CullingSettings {
    pub similarity_threshold: u32,
    pub blur_threshold: f64,
    pub group_similar: bool,
    pub filter_blurry: bool,
    pub analyze_eyes: bool,
    pub auto_assign_stars: bool,
    pub star_mapping: StarMapping,
}

impl Default for CullingSettings {
    fn default() -> Self {
        Self {
            similarity_threshold: 28,
            blur_threshold: 100.0,
            group_similar: true,
            filter_blurry: true,
            analyze_eyes: true,
            auto_assign_stars: true,
            star_mapping: StarMapping::default(),
        }
    }
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ImageAnalysisResult {
    pub path: String,
    pub quality_score: f64,
    pub sharpness_metric: f64,
    pub center_focus_metric: f64,
    pub exposure_metric: f64,
    pub width: u32,
    pub height: u32,
    /// `open`, `closed`, `notApplicable`, or `unknown`.
    pub eye_state: String,
    pub eye_confidence: f32,
    /// Explicitly identifies the deterministic local heuristic when available.
    pub eye_method: String,
    pub category: String,
    pub suggested_rating: u8,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CullGroup {
    pub representative: ImageAnalysisResult,
    pub duplicates: Vec<ImageAnalysisResult>,
}

#[derive(Serialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct CullingSuggestions {
    pub similar_groups: Vec<CullGroup>,
    pub blurry_images: Vec<ImageAnalysisResult>,
    pub retained_images: Vec<ImageAnalysisResult>,
    pub review_images: Vec<ImageAnalysisResult>,
    pub duplicate_images: Vec<ImageAnalysisResult>,
    pub defect_images: Vec<ImageAnalysisResult>,
    pub unknown_images: Vec<ImageAnalysisResult>,
    pub failed_paths: Vec<String>,
    pub star_assignments: HashMap<String, u8>,
    pub eye_analysis_status: String,
}

#[derive(Serialize, Clone)]
struct CullingProgress {
    current: usize,
    total: usize,
    stage: String,
}

struct ImageAnalysisData {
    hash: image_hasher::ImageHash,
    result: ImageAnalysisResult,
}

const WEIGHT_SHARPNESS: f64 = 0.40;
const WEIGHT_CENTER_FOCUS: f64 = 0.35;
const WEIGHT_EXPOSURE: f64 = 0.25;

fn calculate_laplacian_variance(image: &GrayImage) -> f64 {
    let (width, height) = image.dimensions();
    if width < 3 || height < 3 {
        return 0.0;
    }
    let mut laplacian_values = Vec::with_capacity(((width - 2) * (height - 2)) as usize);
    let mut sum = 0.0;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let p_center = image.get_pixel(x, y)[0] as i32;
            let p_north = image.get_pixel(x, y - 1)[0] as i32;
            let p_south = image.get_pixel(x, y + 1)[0] as i32;
            let p_west = image.get_pixel(x - 1, y)[0] as i32;
            let p_east = image.get_pixel(x + 1, y)[0] as i32;
            let value = (p_north + p_south + p_west + p_east - 4 * p_center) as f64;
            laplacian_values.push(value);
            sum += value;
        }
    }
    if laplacian_values.is_empty() {
        return 0.0;
    }
    let mean = sum / laplacian_values.len() as f64;
    laplacian_values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / laplacian_values.len() as f64
}

fn calculate_exposure_metric(image: &GrayImage) -> f64 {
    let histogram = imageproc::stats::histogram(image);
    let total_pixels = (image.width() * image.height()) as f64;
    if total_pixels == 0.0 {
        return 0.0;
    }
    let dark_pixels = histogram.channels[0][0..5].iter().sum::<u32>() as f64;
    let bright_pixels = histogram.channels[0][250..256].iter().sum::<u32>() as f64;
    (1.0 - ((dark_pixels / total_pixels) * 5.0 + (bright_pixels / total_pixels) * 5.0)).max(0.0)
}

fn analyze_image(
    path: &str,
    hasher: &image_hasher::Hasher,
    settings: &crate::app_settings::AppSettings,
) -> Result<ImageAnalysisData, String> {
    const ANALYSIS_DIM: u32 = 720;
    if crate::file_management::is_cloud_placeholder(Path::new(path)) {
        return Err(format!("'{path}' is stored in iCloud and not downloaded"));
    }
    let file_bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    let image = image_loader::load_base_image_from_bytes(&file_bytes, path, true, settings, None)
        .map_err(|error| error.to_string())?;
    let (width, height) = image.dimensions();
    let thumbnail = image.thumbnail(ANALYSIS_DIM, ANALYSIS_DIM);
    let gray_thumbnail = thumbnail.to_luma8();
    let sharpness_metric = calculate_laplacian_variance(&gray_thumbnail);
    let exposure_metric = calculate_exposure_metric(&gray_thumbnail);
    let (thumb_w, thumb_h) = gray_thumbnail.dimensions();
    let center_crop = imageops::crop_imm(
        &gray_thumbnail,
        thumb_w / 4,
        thumb_h / 4,
        thumb_w / 2,
        thumb_h / 2,
    )
    .to_image();
    let center_focus_metric = calculate_laplacian_variance(&center_crop);
    let normalized_sharpness = ((sharpness_metric + 1.0).log10() / 3.5).clamp(0.0, 1.0);
    let normalized_center_focus = ((center_focus_metric + 1.0).log10() / 3.5).clamp(0.0, 1.0);
    let quality_score = normalized_sharpness * WEIGHT_SHARPNESS
        + normalized_center_focus * WEIGHT_CENTER_FOCUS
        + exposure_metric * WEIGHT_EXPOSURE;
    Ok(ImageAnalysisData {
        hash: hasher.hash_image(&thumbnail),
        result: ImageAnalysisResult {
            path: path.to_owned(),
            quality_score,
            sharpness_metric,
            center_focus_metric,
            exposure_metric,
            width,
            height,
            eye_state: "unknown".to_owned(),
            eye_confidence: 0.0,
            eye_method: "unavailable".to_owned(),
            category: "review".to_owned(),
            suggested_rating: 2,
        },
    })
}

fn connected_components(
    item_count: usize,
    mut is_similar: impl FnMut(usize, usize) -> bool,
) -> Vec<Vec<usize>> {
    let mut components = Vec::new();
    let mut processed_indices = vec![false; item_count];
    for i in 0..item_count {
        if processed_indices[i] {
            continue;
        }
        let mut component = vec![i];
        let mut queue = VecDeque::from([i]);
        processed_indices[i] = true;
        while let Some(current) = queue.pop_front() {
            for j in 0..item_count {
                if processed_indices[j] || !is_similar(current, j) {
                    continue;
                }
                processed_indices[j] = true;
                component.push(j);
                queue.push_back(j);
            }
        }
        components.push(component);
    }
    components
}

fn compare_group_candidates(
    left: &ImageAnalysisResult,
    right: &ImageAnalysisResult,
) -> std::cmp::Ordering {
    right
        .quality_score
        .total_cmp(&left.quality_score)
        .then_with(|| left.path.cmp(&right.path))
}

fn encode_face_source(
    path: &str,
    settings: &crate::app_settings::AppSettings,
) -> Result<Vec<u8>, String> {
    let file_bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    let image = image_loader::load_base_image_from_bytes(&file_bytes, path, true, settings, None)
        .map_err(|error| error.to_string())?;
    let bounded = image.thumbnail(1600, 1600).to_rgb8();
    let mut output = std::io::Cursor::new(Vec::new());
    bounded
        .write_to(&mut output, image::ImageFormat::Jpeg)
        .map_err(|error| format!("cannot prepare local face analysis image: {error}"))?;
    Ok(output.into_inner())
}

fn category_rating(category: &str, mapping: &StarMapping) -> u8 {
    let value = match category {
        "retained" => mapping.retained,
        "review" => mapping.review,
        "duplicate" => mapping.duplicate,
        "defect" => mapping.defect,
        _ => mapping.unknown,
    };
    value.min(5)
}

fn is_blurry(result: &ImageAnalysisResult, settings: &CullingSettings) -> bool {
    settings.filter_blurry && result.sharpness_metric < settings.blur_threshold
}

fn culling_category(
    result: &ImageAnalysisResult,
    settings: &CullingSettings,
    is_duplicate: bool,
) -> &'static str {
    if is_duplicate {
        "duplicate"
    } else if settings.analyze_eyes
        && (result.eye_method == "unavailable" || result.eye_state == "unknown")
    {
        "unknown"
    } else if is_blurry(result, settings) || result.eye_state == "closed" {
        "defect"
    } else if result.quality_score >= 0.55 {
        "retained"
    } else {
        "review"
    }
}

fn set_eye_signal(result: &mut ImageAnalysisResult, analysis: &face_processing::LocalFaceAnalysis) {
    if analysis.faces.is_empty() {
        result.eye_state = "notApplicable".to_owned();
        result.eye_method = "yunet-no-face".to_owned();
        return;
    }
    let mut saw_open = false;
    let mut saw_closed = false;
    let mut saw_indeterminate = false;
    let mut confidence = 0.0_f32;
    let mut method = "local-heuristic";
    for face in &analysis.faces {
        confidence = confidence.max(face.eye.confidence);
        method = face.eye.method;
        match face.eye.state {
            face_processing::EyeState::Open => saw_open = true,
            face_processing::EyeState::Closed => saw_closed = true,
            face_processing::EyeState::Unknown | face_processing::EyeState::NotApplicable => {
                saw_indeterminate = true
            }
        }
    }
    result.eye_state = if saw_indeterminate {
        "unknown"
    } else if saw_closed && !saw_open {
        "closed"
    } else if saw_open && !saw_closed {
        "open"
    } else {
        "unknown"
    }
    .to_owned();
    result.eye_confidence = confidence;
    result.eye_method = method.to_owned();
}

#[tauri::command]
pub async fn cull_images(
    paths: Vec<String>,
    settings: CullingSettings,
    app_handle: AppHandle,
    face_state: State<'_, PicPortalState>,
) -> Result<CullingSuggestions, String> {
    if paths.is_empty() {
        return Ok(CullingSuggestions::default());
    }
    let app_settings = load_settings(app_handle.clone()).unwrap_or_default();
    let total_count = paths.len();
    let completed_count = Arc::new(AtomicUsize::new(0));
    let _ = app_handle.emit("culling-start", total_count);
    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::DoubleGradient)
        .hash_size(16, 16)
        .to_hasher();
    let analysis_results: Vec<Result<ImageAnalysisData, (String, String)>> = paths
        .par_iter()
        .map(|path| {
            let completed = completed_count.fetch_add(1, Ordering::Relaxed) + 1;
            let _ = app_handle.emit(
                "culling-progress",
                CullingProgress {
                    current: completed,
                    total: total_count,
                    stage: "Analyzing images locally...".to_owned(),
                },
            );
            analyze_image(path, &hasher, &app_settings).map_err(|error| (path.to_owned(), error))
        })
        .collect();
    let mut successful = Vec::new();
    let mut failed_paths = Vec::new();
    for result in analysis_results {
        match result {
            Ok(data) => successful.push(data),
            Err((path, error)) => {
                log::warn!("Failed to analyze image {path}: {error}");
                failed_paths.push(path);
            }
        }
    }

    let mut eye_status = if settings.analyze_eyes {
        "pending".to_owned()
    } else {
        "disabled".to_owned()
    };
    if settings.analyze_eyes && !successful.is_empty() {
        let mut runtime_guard = face_state
            .face_runtime
            .lock()
            .map_err(|_| "face runtime lock is poisoned".to_owned())?;
        match face_processing::ensure_runtime(&mut runtime_guard, &app_handle) {
            Ok(runtime) => {
                eye_status = "available-local-heuristic".to_owned();
                for data in &mut successful {
                    match encode_face_source(&data.result.path, &app_settings)
                        .and_then(|bytes| runtime.analyze(&bytes))
                    {
                        Ok(analysis) => set_eye_signal(&mut data.result, &analysis),
                        Err(error) => {
                            data.result.eye_method = "unavailable".to_owned();
                            log::debug!(
                                "Eye analysis unavailable for {}: {error}",
                                data.result.path
                            );
                        }
                    }
                }
            }
            Err(error) => {
                eye_status = format!("unavailable: {error}");
                for data in &mut successful {
                    data.result.eye_method = "unavailable".to_owned();
                }
            }
        }
    }

    let _ = app_handle.emit(
        "culling-progress",
        CullingProgress {
            current: total_count,
            total: total_count,
            stage: "Grouping similar images and assigning review buckets...".to_owned(),
        },
    );
    let mut suggestions = CullingSuggestions {
        failed_paths,
        eye_analysis_status: eye_status,
        ..Default::default()
    };
    let mut duplicate_paths = HashSet::new();
    if settings.group_similar {
        for mut group_indices in connected_components(successful.len(), |left, right| {
            successful[left].hash.dist(&successful[right].hash) <= settings.similarity_threshold
        }) {
            if group_indices.len() > 1 {
                group_indices.sort_by(|left, right| {
                    compare_group_candidates(&successful[*left].result, &successful[*right].result)
                });
                let representative = group_indices[0];
                for index in group_indices.iter().skip(1) {
                    duplicate_paths.insert(successful[*index].result.path.clone());
                }
                suggestions.similar_groups.push(CullGroup {
                    representative: successful[representative].result.clone(),
                    duplicates: group_indices
                        .iter()
                        .skip(1)
                        .map(|index| successful[*index].result.clone())
                        .collect(),
                });
            }
        }
    }

    for data in &mut successful {
        let category = culling_category(
            &data.result,
            &settings,
            duplicate_paths.contains(&data.result.path),
        );
        data.result.category = category.to_owned();
        data.result.suggested_rating = category_rating(category, &settings.star_mapping);
        if settings.auto_assign_stars {
            suggestions
                .star_assignments
                .insert(data.result.path.clone(), data.result.suggested_rating);
        }
        match category {
            "retained" => suggestions.retained_images.push(data.result.clone()),
            "review" => suggestions.review_images.push(data.result.clone()),
            "duplicate" => suggestions.duplicate_images.push(data.result.clone()),
            "unknown" => suggestions.unknown_images.push(data.result.clone()),
            _ => suggestions.defect_images.push(data.result.clone()),
        }
        if is_blurry(&data.result, &settings) {
            suggestions.blurry_images.push(data.result.clone());
        }
    }
    suggestions
        .retained_images
        .sort_by(|left, right| right.quality_score.total_cmp(&left.quality_score));
    suggestions
        .review_images
        .sort_by(|left, right| right.quality_score.total_cmp(&left.quality_score));
    suggestions
        .duplicate_images
        .sort_by(|left, right| left.path.cmp(&right.path));
    suggestions
        .defect_images
        .sort_by(|left, right| left.path.cmp(&right.path));
    suggestions
        .unknown_images
        .sort_by(|left, right| left.path.cmp(&right.path));
    suggestions
        .blurry_images
        .sort_by(|left, right| left.sharpness_metric.total_cmp(&right.sharpness_metric));
    let _ = app_handle.emit("culling-complete", &suggestions);
    Ok(suggestions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analysis_result(path: &str, quality_score: f64) -> ImageAnalysisResult {
        ImageAnalysisResult {
            path: path.to_owned(),
            quality_score,
            sharpness_metric: 0.0,
            center_focus_metric: 0.0,
            exposure_metric: 0.0,
            width: 1,
            height: 1,
            eye_state: "notApplicable".to_owned(),
            eye_confidence: 0.0,
            eye_method: "disabled".to_owned(),
            category: "review".to_owned(),
            suggested_rating: 0,
        }
    }

    fn eye_settings(analyze_eyes: bool) -> CullingSettings {
        CullingSettings {
            analyze_eyes,
            filter_blurry: false,
            group_similar: false,
            ..Default::default()
        }
    }

    fn face(state: face_processing::EyeState) -> face_processing::LocalFace {
        face_processing::LocalFace {
            bbox: face_processing::FaceBox {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
            confidence: 1.0,
            embedding: Vec::new(),
            thumbnail: Vec::new(),
            thumbnail_sha256: String::new(),
            eye: face_processing::EyeAssessment {
                state,
                confidence: 0.5,
                method: "local-heuristic",
            },
        }
    }

    fn face_analysis(states: Vec<face_processing::EyeState>) -> face_processing::LocalFaceAnalysis {
        face_processing::LocalFaceAnalysis {
            source_sha256: String::new(),
            source_width: 1,
            source_height: 1,
            faces: states.into_iter().map(face).collect(),
            model_id: String::new(),
            model_digest: String::new(),
            pipeline_version: String::new(),
            embedding_dimension: 0,
            detector_input: 0,
        }
    }

    fn category_and_rating(
        result: &ImageAnalysisResult,
        settings: &CullingSettings,
    ) -> (&'static str, u8) {
        let category = culling_category(result, settings, false);
        (category, category_rating(category, &settings.star_mapping))
    }

    #[test]
    fn star_mapping_is_clamped_to_the_rapidraw_range() {
        let mapping = StarMapping {
            retained: 9,
            review: 4,
            duplicate: 2,
            defect: 1,
            unknown: 3,
        };
        assert_eq!(category_rating("retained", &mapping), 5);
        assert_eq!(category_rating("review", &mapping), 4);
        assert_eq!(category_rating("duplicate", &mapping), 2);
        assert_eq!(category_rating("defect", &mapping), 1);
        assert_eq!(category_rating("unknown", &mapping), 3);
    }

    #[test]
    fn similarity_groups_include_lower_index_neighbors_discovered_later() {
        let edges = [(0, 2), (2, 0), (2, 1), (1, 2)];
        let groups = connected_components(3, |left, right| edges.contains(&(left, right)));
        assert_eq!(groups, vec![vec![0, 2, 1]]);
    }

    #[test]
    fn equal_quality_representatives_are_independent_of_input_order() {
        let first = analysis_result("a.jpg", 0.75);
        let second = analysis_result("b.jpg", 0.75);
        let mut forward = [&first, &second];
        let mut reversed = [&second, &first];

        forward.sort_by(|left, right| compare_group_candidates(left, right));
        reversed.sort_by(|left, right| compare_group_candidates(left, right));

        assert_eq!(forward[0].path, "a.jpg");
        assert_eq!(reversed[0].path, "a.jpg");
    }

    #[test]
    fn eye_outcome_indeterminate_faces_are_unknown() {
        let settings = eye_settings(true);
        for states in [
            vec![
                face_processing::EyeState::Closed,
                face_processing::EyeState::Unknown,
            ],
            vec![
                face_processing::EyeState::Open,
                face_processing::EyeState::Closed,
            ],
        ] {
            let mut result = analysis_result("faces.jpg", 0.75);
            set_eye_signal(&mut result, &face_analysis(states));

            assert_eq!(result.eye_state, "unknown");
            assert_eq!(
                category_and_rating(&result, &settings),
                ("unknown", settings.star_mapping.unknown)
            );
        }
    }

    #[test]
    fn eye_outcome_conclusive_faces_preserve_categories() {
        let settings = eye_settings(true);
        let mut open = analysis_result("open.jpg", 0.75);
        set_eye_signal(
            &mut open,
            &face_analysis(vec![
                face_processing::EyeState::Open,
                face_processing::EyeState::Open,
            ]),
        );
        let mut closed = analysis_result("closed.jpg", 0.75);
        set_eye_signal(
            &mut closed,
            &face_analysis(vec![
                face_processing::EyeState::Closed,
                face_processing::EyeState::Closed,
            ]),
        );

        assert_eq!(
            category_and_rating(&open, &settings),
            ("retained", settings.star_mapping.retained)
        );
        assert_eq!(
            category_and_rating(&closed, &settings),
            ("defect", settings.star_mapping.defect)
        );
    }

    #[test]
    fn eye_outcome_analysis_failure_is_unknown() {
        let settings = eye_settings(true);
        let mut result = analysis_result("failed.jpg", 0.75);
        result.eye_state = "unknown".to_owned();
        result.eye_method = "unavailable".to_owned();

        assert_eq!(
            category_and_rating(&result, &settings),
            ("unknown", settings.star_mapping.unknown)
        );
    }

    #[test]
    fn eye_outcome_disabled_analysis_uses_quality() {
        let settings = eye_settings(false);
        let mut result = analysis_result("disabled.jpg", 0.75);
        result.eye_state = "unknown".to_owned();
        result.eye_method = "unavailable".to_owned();

        assert_eq!(
            category_and_rating(&result, &settings),
            ("retained", settings.star_mapping.retained)
        );
    }
}
