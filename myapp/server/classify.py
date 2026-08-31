#!/usr/bin/env python3
"""Classify an image into one of nine trained visual categories.

Feature layout matches training (776 values):
  mean, std, min, max, edge count, HSV means, then 256-bin R/G/B histograms.
Weights live in data/theta_List.csv (one-vs-rest logistic regression).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
THETA_PATH = ROOT / "data" / "theta_List.csv"

CLASSES = [
    {"id": 1, "label": "Attack on Titan", "folder": "attackOnTitan_fotos"},
    {"id": 2, "label": "Death Note", "folder": "deathNote_fotos"},
    {"id": 3, "label": "Evangelion", "folder": "Evangelion"},
    {"id": 4, "label": "Demon Slayer", "folder": "KimetsuNoYaiba"},
    {"id": 5, "label": "The Lord of the Rings", "folder": "LOTR"},
    {"id": 6, "label": "NBA", "folder": "NBA"},
    {"id": 7, "label": "One Piece", "folder": "onePiece_fotos"},
    {"id": 8, "label": "Sneakers", "folder": "SNKRS"},
    {"id": 9, "label": "Star Wars", "folder": "StarWars"},
]


def load_theta(path: Path = THETA_PATH) -> np.ndarray:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = [part for part in line.strip().split(",") if part != ""]
            if parts:
                rows.append(parts)
    theta = np.array(rows, dtype=float)
    if theta.ndim != 2 or theta.shape[0] != len(CLASSES):
        raise ValueError(f"Unexpected weight matrix shape: {theta.shape}")
    return theta


def extract_features(img: np.ndarray) -> np.ndarray:
    """Build the same 776-d vector used to train theta_List.csv."""
    if img is None:
        raise ValueError("Could not read image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = cv2.mean(gray)[0]
    std_dev = float(cv2.meanStdDev(gray)[1][0][0])
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    n_edges = float(cv2.countNonZero(cv2.Canny(gray, 50, 150)))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = (float(v) for v in cv2.mean(hsv)[:3])

    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

    stats = np.array(
        [mean, std_dev, min_val, max_val, n_edges, hue, sat, val],
        dtype=float,
    )
    return np.concatenate([stats, hist_r, hist_g, hist_b])


def predict(features: np.ndarray, theta: np.ndarray) -> dict:
    logits = features @ theta.T
    scale = float(np.std(logits)) or 1.0
    shifted = (logits - np.max(logits)) / scale
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)

    idx = int(np.argmax(logits))
    scores = [
        {
            "id": item["id"],
            "label": item["label"],
            "score": float(probs[item["id"] - 1]),
        }
        for item in CLASSES
    ]
    scores.sort(key=lambda row: row["score"], reverse=True)

    return {
        "ok": True,
        "classId": CLASSES[idx]["id"],
        "label": CLASSES[idx]["label"],
        "confidence": float(probs[idx]),
        "scores": scores,
    }


def classify_image(path: str | Path, theta: np.ndarray | None = None) -> dict:
    image_path = Path(path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to load image: {image_path}")

    if theta is None:
        theta = load_theta()

    features = extract_features(img)
    if features.shape[0] != theta.shape[1]:
        raise ValueError(
            f"Feature size {features.shape[0]} does not match model {theta.shape[1]}"
        )

    result = predict(features, theta)
    result["filename"] = image_path.name
    return result


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "Usage: classify.py <image-path>"}))
        return 1
    try:
        print(json.dumps(classify_image(sys.argv[1])), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 — CLI must always emit JSON
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
