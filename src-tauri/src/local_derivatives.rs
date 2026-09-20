//! Local PicPortal delivery derivatives.
//!
//! The exported image remains the upload source.  This module only creates the
//! two WebP delivery objects required by PicPortal and never replaces the
//! source file.  Decoding and EXIF orientation happen before dimensions are
//! recorded so the manifest describes visual dimensions.

use std::io::Cursor;

use image::{
    DynamicImage, GenericImageView, ImageDecoder, ImageFormat, ImageReader, Limits,
    imageops::FilterType,
};
use sha2::{Digest, Sha256};

pub const MAX_SOURCE_BYTES: u64 = 50 * 1024 * 1024;
pub const MAX_SOURCE_PIXELS: u64 = 40_000_000;
pub const PREVIEW_MAX_SIDE: u32 = 2_200;
pub const PREVIEW_QUALITY: f32 = 82.0;
pub const THUMBNAIL_MAX_SIDE: u32 = 640;
pub const THUMBNAIL_QUALITY: f32 = 76.0;
const MAX_SOURCE_DIMENSION: u32 = 16_384;
const MAX_DECODE_ALLOC: u64 = 256 * 1024 * 1024;
const MAX_DERIVATIVE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug)]
pub struct LocalDerivative {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub sha256: String,
}

#[derive(Debug)]
pub struct LocalDerivatives {
    pub source_sha256: String,
    pub source_width: u32,
    pub source_height: u32,
    pub preview: LocalDerivative,
    pub thumbnail: LocalDerivative,
}

#[derive(Debug)]
pub struct DecodedSource {
    pub source_sha256: String,
    pub width: u32,
    pub height: u32,
    pub image: DynamicImage,
}

/// Decode a supported exported image and apply its EXIF orientation.
pub fn decode_source(source: &[u8]) -> Result<DecodedSource, String> {
    let source_len = u64::try_from(source.len()).map_err(|_| "source is too large".to_owned())?;
    if source_len == 0 || source_len > MAX_SOURCE_BYTES {
        return Err(format!(
            "source must be between 1 and {MAX_SOURCE_BYTES} bytes"
        ));
    }

    let mut reader = ImageReader::new(Cursor::new(source))
        .with_guessed_format()
        .map_err(|error| format!("cannot identify exported image: {error}"))?;
    let format = reader
        .format()
        .ok_or_else(|| "exported image has no recognised format".to_owned())?;
    if !matches!(
        format,
        ImageFormat::Jpeg | ImageFormat::Png | ImageFormat::WebP
    ) {
        return Err("PicPortal local delivery supports JPEG, PNG and WebP exports".to_owned());
    }
    if matches!(format, ImageFormat::Png) && !is_static_png(source) {
        return Err(
            "animated or malformed PNG exports are not supported by local PicPortal processing"
                .to_owned(),
        );
    }
    if matches!(format, ImageFormat::WebP) && !is_static_webp(source) {
        return Err(
            "animated or malformed WebP exports are not supported by local PicPortal processing"
                .to_owned(),
        );
    }

    let mut limits = Limits::default();
    limits.max_alloc = Some(MAX_DECODE_ALLOC);
    limits.max_image_width = Some(MAX_SOURCE_DIMENSION);
    limits.max_image_height = Some(MAX_SOURCE_DIMENSION);
    reader.limits(limits);
    let mut decoder = reader
        .into_decoder()
        .map_err(|error| format!("cannot decode exported image: {error}"))?;
    let (raw_width, raw_height) = decoder.dimensions();
    let raw_pixels = u64::from(raw_width) * u64::from(raw_height);
    if raw_width == 0
        || raw_height == 0
        || raw_width.max(raw_height) > MAX_SOURCE_DIMENSION
        || raw_pixels > MAX_SOURCE_PIXELS
    {
        return Err(format!(
            "source exceeds the {MAX_SOURCE_PIXELS}-pixel local limit"
        ));
    }
    let orientation = decoder
        .orientation()
        .map_err(|error| format!("cannot read EXIF orientation: {error}"))?;
    let mut image = DynamicImage::from_decoder(decoder)
        .map_err(|error| format!("cannot materialize exported image: {error}"))?;
    if image.color().has_alpha() && image.pixels().any(|(_, _, pixel)| pixel.0[3] != 255) {
        return Err(
            "transparent exports are not supported by local PicPortal processing".to_owned(),
        );
    }
    image.apply_orientation(orientation);
    let (width, height) = image.dimensions();
    Ok(DecodedSource {
        source_sha256: sha256_hex(source),
        width,
        height,
        image,
    })
}

/// Generate the preview (2200 px / quality 82) and thumbnail (640 px / quality 76).
pub fn generate_local_derivatives(source: &[u8]) -> Result<LocalDerivatives, String> {
    let decoded = decode_source(source)?;
    let preview = make_derivative(&decoded.image, PREVIEW_MAX_SIDE, PREVIEW_QUALITY)?;
    let thumbnail = make_derivative(&decoded.image, THUMBNAIL_MAX_SIDE, THUMBNAIL_QUALITY)?;
    Ok(LocalDerivatives {
        source_sha256: decoded.source_sha256,
        source_width: decoded.width,
        source_height: decoded.height,
        preview,
        thumbnail,
    })
}

pub fn expected_dimensions(width: u32, height: u32, max_side: u32) -> (u32, u32) {
    let largest = width.max(height);
    if largest <= max_side {
        return (width, height);
    }
    (
        scale_dimension(width, largest, max_side),
        scale_dimension(height, largest, max_side),
    )
}

fn scale_dimension(value: u32, largest: u32, max_side: u32) -> u32 {
    let numerator = u64::from(value) * u64::from(max_side);
    let rounded = (numerator + u64::from(largest) / 2) / u64::from(largest);
    u32::try_from(rounded.max(1)).expect("delivery dimension fits in u32")
}

fn make_derivative(
    image: &DynamicImage,
    max_side: u32,
    quality: f32,
) -> Result<LocalDerivative, String> {
    let (width, height) = image.dimensions();
    let (target_width, target_height) = expected_dimensions(width, height, max_side);
    let resized = if (width, height) == (target_width, target_height) {
        image.clone()
    } else {
        image.resize_exact(target_width, target_height, FilterType::Lanczos3)
    };
    let rgb = resized.to_rgb8();
    let bytes = webp::Encoder::from_rgb(&rgb, target_width, target_height)
        .encode(quality)
        .to_vec();
    if bytes.len() > MAX_DERIVATIVE_BYTES {
        return Err("generated WebP exceeds the local derivative size limit".to_owned());
    }
    Ok(LocalDerivative {
        sha256: sha256_hex(&bytes),
        bytes,
        width: target_width,
        height: target_height,
    })
}

fn is_static_png(source: &[u8]) -> bool {
    if source.len() < 33 {
        return false;
    }
    let mut offset = 8_usize;
    while offset + 12 <= source.len() {
        let size = u32::from_be_bytes(
            source[offset..offset + 4]
                .try_into()
                .expect("four-byte PNG chunk size"),
        ) as usize;
        let Some(end) = offset
            .checked_add(12)
            .and_then(|value| value.checked_add(size))
        else {
            return false;
        };
        if end > source.len() {
            return false;
        }
        let chunk = &source[offset + 4..offset + 8];
        if matches!(chunk, b"acTL" | b"fcTL" | b"fdAT") {
            return false;
        }
        if chunk == b"IEND" {
            return end == source.len();
        }
        offset = end;
    }
    false
}

fn is_static_webp(source: &[u8]) -> bool {
    if source.len() < 20 || &source[0..4] != b"RIFF" || &source[8..12] != b"WEBP" {
        return false;
    }
    let mut offset = 12_usize;
    while offset + 8 <= source.len() {
        let chunk = &source[offset..offset + 4];
        let size = u32::from_le_bytes(
            source[offset + 4..offset + 8]
                .try_into()
                .expect("four-byte WebP size"),
        ) as usize;
        let Some(payload_end) = offset
            .checked_add(8)
            .and_then(|value| value.checked_add(size))
        else {
            return false;
        };
        let Some(padded_end) = payload_end.checked_add(size & 1) else {
            return false;
        };
        if padded_end > source.len() {
            return false;
        }
        if chunk == b"ANIM" || chunk == b"ANMF" {
            return false;
        }
        if chunk == b"VP8X" && (size == 0 || source[offset + 8] & 0x02 != 0) {
            return false;
        }
        offset = padded_end;
    }
    offset == source.len()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb, Rgba};

    fn jpeg(width: u32, height: u32) -> Vec<u8> {
        let image =
            DynamicImage::ImageRgb8(ImageBuffer::from_pixel(width, height, Rgb([32, 96, 180])));
        let mut bytes = Cursor::new(Vec::new());
        image.write_to(&mut bytes, ImageFormat::Jpeg).expect("jpeg");
        bytes.into_inner()
    }

    fn transparent_png() -> Vec<u8> {
        let image = DynamicImage::ImageRgba8(ImageBuffer::from_pixel(1, 1, Rgba([32, 96, 180, 0])));
        let mut bytes = Cursor::new(Vec::new());
        image.write_to(&mut bytes, ImageFormat::Png).expect("png");
        bytes.into_inner()
    }

    #[test]
    fn delivery_dimensions_never_enlarge() {
        assert_eq!(expected_dimensions(100, 40, PREVIEW_MAX_SIDE), (100, 40));
        assert_eq!(
            expected_dimensions(4_000, 2_000, PREVIEW_MAX_SIDE),
            (2_200, 1_100)
        );
        assert_eq!(
            expected_dimensions(2_000, 4_000, THUMBNAIL_MAX_SIDE),
            (320, 640)
        );
    }

    #[test]
    fn generated_webp_contract_contains_visual_dimensions_and_hashes() {
        let derivatives = generate_local_derivatives(&jpeg(4_000, 2_000)).expect("derivatives");
        assert_eq!(
            (derivatives.source_width, derivatives.source_height),
            (4_000, 2_000)
        );
        assert_eq!(
            (derivatives.preview.width, derivatives.preview.height),
            (2_200, 1_100)
        );
        assert_eq!(
            (derivatives.thumbnail.width, derivatives.thumbnail.height),
            (640, 320)
        );
        assert!(derivatives.preview.bytes.starts_with(b"RIFF"));
        assert!(derivatives.thumbnail.bytes.starts_with(b"RIFF"));
        assert!(is_static_webp(&derivatives.preview.bytes));
        assert!(is_static_webp(&derivatives.thumbnail.bytes));
        assert_eq!(
            derivatives.preview.sha256,
            sha256_hex(&derivatives.preview.bytes)
        );
    }

    #[test]
    fn unsupported_format_is_rejected_before_network_work() {
        assert!(decode_source(b"not-an-image").is_err());
    }

    #[test]
    fn unsupported_exports_report_local_rejection() {
        let png = transparent_png();
        assert_eq!(
            decode_source(&png).expect_err("transparent PNG must be rejected"),
            "transparent exports are not supported by local PicPortal processing"
        );
        assert_eq!(
            decode_source(&png[..20]).expect_err("malformed PNG must be rejected"),
            "animated or malformed PNG exports are not supported by local PicPortal processing"
        );

        let webp = webp::Encoder::from_rgb(&[32, 96, 180], 1, 1)
            .encode(75.0)
            .to_vec();
        assert_eq!(
            decode_source(&webp[..20]).expect_err("malformed WebP must be rejected"),
            "animated or malformed WebP exports are not supported by local PicPortal processing"
        );
    }
}
