"""Exercise the packaged app with no dependency on the original workspace."""
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class ReviewWebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.dataset = Path(cls.temp.name) / "dataset"
        cls.output = Path(cls.temp.name) / "results"
        shutil.copytree(ROOT / "demo", cls.dataset)
        cls.old_env = {key: os.environ.get(key) for key in
                       ["REVIEW_DATA_ROOT", "REVIEW_OUTPUT_DIR", "REVIEW_CONFIG"]}
        os.environ["REVIEW_DATA_ROOT"] = str(cls.dataset)
        os.environ["REVIEW_OUTPUT_DIR"] = str(cls.output)
        os.environ["REVIEW_CONFIG"] = str(ROOT / "config.yaml")
        cls.web = importlib.import_module("review_web")

    @classmethod
    def tearDownClass(cls):
        for key, value in cls.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        cls.temp.cleanup()

    def setUp(self):
        shutil.rmtree(self.output, ignore_errors=True)
        self.source_before = (self.dataset / "output/samples.json").read_bytes()
        self.client = TestClient(self.web.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.assertEqual(self.source_before, (self.dataset / "output/samples.json").read_bytes())

    def test_ui_and_images_without_metadata_exposure(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        summary = self.client.get("/api/samples").json()
        self.assertEqual(summary["total"], 1)
        sample = self.client.get("/api/sample/demo_001").json()
        self.assertTrue(sample["question_suggestions"])
        for image in sample["ground_truth"] + sample["non_gt_images"]:
            response = self.client.get(image["img_url"])
            self.assertEqual(response.status_code, 200, image["img_url"])
            self.assertEqual(response.headers["content-type"], "image/jpeg")
        crop = sample["ground_truth"][0]["gt_bboxes"][0]["crop_url"]
        self.assertEqual(self.client.get(crop).status_code, 200)
        for path in ["/data/output/samples.json", "/data/intermediate/enriched_attributes.json",
                     "/reviews/sample_review_state.json"]:
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_save_and_restart_restore_results(self):
        response = self.client.post("/api/sample/demo_001/save", json={"status": "pass"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["review"]["status"], "pass")
        reviews = json.loads((self.output / "sample_review_state.json").read_text())
        self.assertEqual(reviews["demo_001"]["status"], "pass")
        samples = json.loads((self.output / "samples_reviewed.json").read_text())
        self.assertEqual(len(samples[0]["pools"]["N_20"]["images"]), 20)
        self.web.init_data()
        restored = self.client.get("/api/sample/demo_001").json()
        self.assertEqual(restored["review"]["status"], "pass")
        self.assertEqual(len(self.web.STATE["raw_bboxes"]),
                         sum(len(items) for items in self.web.STATE["raw_img_bboxes"].values()))

    def test_replace_non_gt_and_reset(self):
        sample = self.client.get("/api/sample/demo_001").json()
        old = sample["non_gt_images"][0]["image_id"]
        pool = self.client.get("/api/sample/demo_001/replacement_pool").json()
        candidates = pool["other_image_candidates"] + pool["hard_distractor_candidates"]
        self.assertTrue(candidates)
        candidate = candidates[0]
        response = self.client.post("/api/sample/demo_001/replace_non_gt", json={
            "old_image_id": old, "new_image_id": candidate["image_id"],
            "new_role": candidate["role"], "new_reason": candidate["reason"],
        })
        self.assertEqual(response.status_code, 200, response.text)
        ids = {item["image_id"] for item in response.json()["non_gt_images"]}
        self.assertIn(candidate["image_id"], ids)
        self.assertNotIn(old, ids)
        response = self.client.post("/api/sample/demo_001/reset")
        self.assertEqual(response.json()["review"]["status"], "pending")
        self.assertEqual(json.loads((self.output / "sample_review_state.json").read_text()), {})

    def test_reject_gt_and_export(self):
        response = self.client.post("/api/sample/demo_001/save", json={"no_valid_gt_images": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["review"]["status"], "no_valid_gt")
        export = self.client.get("/api/export").json()
        self.assertEqual(export["total_original_samples"], 1)
        self.assertEqual(export["total_valid_samples"], 0)

    def test_unknown_sample(self):
        self.assertEqual(self.client.get("/api/sample/missing").status_code, 404)


if __name__ == "__main__":
    unittest.main()
