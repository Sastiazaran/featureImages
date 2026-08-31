#!/usr/bin/env python3
"""Sanity checks for Frame Lab feature extraction and classification."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

SERVER = Path(__file__).resolve().parent
REPO = SERVER.parents[1]
SAMPLE = (
    REPO
    / "imageClassifier"
    / "fotosGabriel"
    / "attackOnTitan_fotos"
    / "AoT (1).png"
)

sys.path.insert(0, str(SERVER))
from classify import CLASSES, classify_image, extract_features, load_theta  # noqa: E402


def load_training_matrix():
    features_path = SERVER / "features.csv"
    hist_r = SERVER / "featuresHistR.csv"
    hist_g = SERVER / "featuresHistG.csv"
    hist_b = SERVER / "featuresHistB.csv"

    rows = []
    with open(features_path, encoding="latin-1") as handle:
        next(handle)
        for line in handle:
            rows.append(line.split(",")[2:10])

    for path in (hist_r, hist_g, hist_b):
        with open(path, encoding="utf-8") as handle:
            next(handle)
            for index, line in enumerate(handle):
                parts = line.split(",")
                rows[index].extend(parts[2:258])

    labels = []
    mapping = {item["folder"]: item["id"] for item in CLASSES}
    with open(features_path, encoding="latin-1") as handle:
        next(handle)
        for line in handle:
            folder = line.split(",")[1]
            labels.append(mapping[folder])

    return np.array(rows, dtype=float), np.array(labels)


class ClassifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.theta = load_theta()

    def test_theta_shape(self):
        self.assertEqual(self.theta.shape, (9, 776))

    def test_sample_image_exists(self):
        self.assertTrue(SAMPLE.exists(), f"Missing sample still: {SAMPLE}")

    def test_extracted_features_match_csv(self):
        import cv2

        image = cv2.imread(str(SAMPLE), cv2.IMREAD_COLOR)
        extracted = extract_features(image)
        self.assertEqual(extracted.shape, (776,))

        with open(SERVER / "features.csv", encoding="latin-1") as handle:
            next(handle)
            stats = np.array(next(handle).split(",")[2:10], dtype=float)
        np.testing.assert_allclose(extracted[:8], stats, rtol=1e-6, atol=1e-6)

        with open(SERVER / "featuresHistR.csv", encoding="utf-8") as handle:
            next(handle)
            hist = np.array(next(handle).split(",")[2:258], dtype=float)
        np.testing.assert_allclose(extracted[8:264], hist, rtol=1e-5, atol=1e-5)

    def test_known_attack_on_titan_still(self):
        result = classify_image(SAMPLE, theta=self.theta)
        self.assertTrue(result["ok"])
        self.assertEqual(result["classId"], 1)
        self.assertEqual(result["label"], "Attack on Titan")

    def test_cli_emits_json(self):
        completed = subprocess.run(
            [sys.executable, str(SERVER / "classify.py"), str(SAMPLE)],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["classId"], 1)

    def test_training_argmax_accuracy(self):
        features, labels = load_training_matrix()
        logits = features @ self.theta.T
        predicted = logits.argmax(axis=1) + 1
        accuracy = float((predicted == labels).mean())
        self.assertGreaterEqual(accuracy, 0.75)


if __name__ == "__main__":
    unittest.main()
