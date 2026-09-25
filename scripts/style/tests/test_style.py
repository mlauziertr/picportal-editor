"""Unit tests for the style model pipeline.

    python3 -m unittest discover -s scripts/style/tests -t .
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.style import extract, features, synth, targets, train
from scripts.style.models import ClusterMean, Knn, Mlp, Ridge
from scripts.style.schema import FEATURE_COUNT, FEATURE_NAMES, OUTPUT_KEYS, clamp_adjustments

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "models" / "style" / "fixtures"


class FeatureTests(unittest.TestCase):
    def test_box_preview_averages_integer_blocks(self):
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        image[:, 4:] = 255
        preview = features.box_preview(image, size=2)
        np.testing.assert_allclose(preview[..., 0], [[0, 1], [0, 1]])

    def test_box_preview_upsamples_small_images_without_empty_blocks(self):
        image = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        preview = features.box_preview(image, size=4)
        self.assertEqual(preview.shape, (4, 4, 3))
        np.testing.assert_allclose(preview[0, :, 0], [0, 0, 1, 1])

    def test_feature_vector_shape_and_names(self):
        preview = np.full((64, 64, 3), 0.5)
        values = features.compute_features(preview, features.Exif(400, 1 / 60, 35))
        self.assertEqual(values.shape, (FEATURE_COUNT,))
        self.assertEqual(len(FEATURE_NAMES), FEATURE_COUNT)
        self.assertAlmostEqual(values[FEATURE_NAMES.index("log2_iso")], 2.0)
        self.assertEqual(values[FEATURE_NAMES.index("exif_present")], 1.0)
        missing = features.compute_features(preview, features.Exif())
        self.assertEqual(missing[FEATURE_NAMES.index("exif_present")], 0.0)

    def test_raw_extensions_match_formats_rs(self):
        source = (REPO / "src-tauri" / "src" / "formats.rs").read_text()
        block = source.split("pub const RAW_EXTENSIONS")[1].split("];")[0]
        self.assertEqual(set(re.findall(r'\("([a-z0-9]+)"', block)), set(features.RAW_EXTENSIONS))

    def test_parity_fixtures_are_reproducible(self):
        expected = json.loads((FIXTURES / "parity.expected.json").read_text())
        for entry in expected["images"]:
            actual = features.features_for_path(FIXTURES / entry["file"])
            np.testing.assert_allclose(actual, entry["features"], atol=1e-9, err_msg=entry["file"])

    def test_raw_preview_and_exif_are_read_from_tiff_containers(self):
        rgb, exif = features.load_source(FIXTURES / "parity-preview.dng")
        self.assertEqual(exif, features.Exif(800.0, 1 / 125, 50.0))
        self.assertEqual(rgb.shape, (240, 160, 3))  # orientation 8 rotates the 240x160 preview


class TargetTests(unittest.TestCase):
    def test_lightroom_fixture_matches_expected_conversion(self):
        converted = targets.adjustments_from_xmp((FIXTURES / "lightroom-basic.xmp").read_text())
        expected = json.loads((FIXTURES / "lightroom-basic.expected.json").read_text())
        for key in OUTPUT_KEYS:
            self.assertAlmostEqual(converted[key], expected[key], places=9)
        self.assertAlmostEqual(converted["shadows"], 45.0)
        self.assertAlmostEqual(converted["tint"], 8.0)
        self.assertGreater(converted["temperature"], 0)  # 5800 K over a 5200 K shot renders warmer

    def test_xmp_without_develop_settings_is_not_an_edit(self):
        self.assertIsNone(targets.adjustments_from_xmp('<x:xmpmeta xmp:Rating="3"/>'))

    def test_element_form_xmp_is_supported(self):
        content = "<crs:ProcessVersion>15.4</crs:ProcessVersion><crs:Exposure2012>+1.25</crs:Exposure2012>"
        self.assertEqual(targets.adjustments_from_xmp(content)["exposure"], 1.25)

    def test_rrdata_requires_a_moved_basic_slider(self):
        self.assertIsNone(targets.adjustments_from_rrdata('{"adjustments": {"clarity": 20}}'))
        self.assertIsNone(targets.adjustments_from_rrdata('{"adjustments": null}'))
        edit = targets.adjustments_from_rrdata('{"adjustments": {"exposure": 0.4, "contrast": "bad"}}')
        self.assertEqual(edit["exposure"], 0.4)
        self.assertEqual(edit["contrast"], 0.0)

    def test_clamp_rounds_like_the_sliders(self):
        clamped = clamp_adjustments({key: 123.456 for key in OUTPUT_KEYS})
        self.assertEqual(clamped["exposure"], 5.0)
        self.assertEqual(clamped["contrast"], 100.0)


class ModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.features = rng.normal(size=(80, FEATURE_COUNT))
        self.targets = self.features[:, : len(OUTPUT_KEYS)] * 10 + rng.normal(size=(80, len(OUTPUT_KEYS)))

    def test_onnx_graphs_match_numpy_predictions(self):
        try:
            import onnxruntime
        except ImportError:
            self.skipTest("onnxruntime not installed")
        probe = self.features[:7]
        for model in (Ridge(1.0), Knn(5), Mlp(16, epochs=50), ClusterMean(3)):
            model.fit(self.features, self.targets, seed=0)
            session = onnxruntime.InferenceSession(model.to_onnx().SerializeToString())
            (actual,) = session.run(None, {"features": probe.astype(np.float32)})
            np.testing.assert_allclose(actual, model.predict(probe), rtol=1e-3, atol=1e-2, err_msg=model.kind)

    def test_cluster_mean_predicts_group_averages(self):
        model = ClusterMean(2).fit(self.features, self.targets, seed=0)
        self.assertEqual(sum(model.sizes), len(self.features))
        predictions = model.predict(self.features)
        self.assertEqual(len(np.unique(predictions.round(6), axis=0)), 2)


class PipelineTests(unittest.TestCase):
    def test_synthetic_corpus_trains_a_model_that_beats_the_baseline(self):
        with tempfile.TemporaryDirectory() as scratch:
            corpus, out = Path(scratch) / "corpus", Path(scratch) / "model"
            truth = synth.generate(corpus, count=90, seed=5)
            self.assertEqual(train.main(["--corpus", str(corpus), "--out", str(out), "--workers", "1"]), 0)
            manifest = json.loads((out / "manifest.json").read_text())
            report = json.loads((out / "report.json").read_text())
            artifact = manifest["artifacts"][0]
            self.assertEqual(train.sha256(out / artifact["filename"]), artifact["sha256"])
            self.assertFalse(manifest["fallback"])
            self.assertTrue(report["evaluation"]["beatsBaseline"])
            self.assertNotIn(scratch, (out / "manifest.json").read_text() + (out / "report.json").read_text())
            edited = sum(1 for record in truth["images"].values() if record.get("edited", True))
            self.assertEqual(report["samples"], edited)

    def test_small_corpus_exports_the_cluster_mean_fallback(self):
        with tempfile.TemporaryDirectory() as scratch:
            corpus, out = Path(scratch) / "corpus", Path(scratch) / "model"
            synth.generate(corpus, count=30, seed=6)
            dataset = extract.build_dataset([corpus], workers=1)
            self.assertLess(dataset["summary"]["samples"], 50)
            model, report = train.evaluate(dataset, seed=0, truth_path=None)
            manifest = train.export(model, report, dataset, out, seed=0)
            self.assertTrue(manifest["fallback"])
            self.assertEqual(manifest["kind"], "cluster_mean")

    def test_dataset_round_trip(self):
        with tempfile.TemporaryDirectory() as scratch:
            corpus = Path(scratch) / "corpus"
            synth.generate(corpus, count=12, seed=8)
            (corpus / "broken.jpg").write_bytes(b"not a jpeg")
            (corpus / "broken.jpg.rrdata").write_text('{"adjustments": {"exposure": 1}}')
            dataset = extract.build_dataset([corpus], workers=1)
            self.assertEqual(dataset["summary"]["skipped"].get("unreadable"), 1)
            path = Path(scratch) / "dataset.npz"
            extract.save_dataset(dataset, path)
            loaded = extract.load_dataset(path)
            np.testing.assert_array_equal(loaded["features"], dataset["features"])
            self.assertEqual(loaded["ids"], dataset["ids"])


if __name__ == "__main__":
    unittest.main()
