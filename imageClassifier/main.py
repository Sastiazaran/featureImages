"""Extract histogram features from the labeled stills in fotosGabriel."""

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PHOTOS = ROOT / "fotosGabriel"

FOLDERS = [
    "attackOnTitan_fotos",
    "deathNote_fotos",
    "Evangelion",
    "KimetsuNoYaiba",
    "LOTR",
    "NBA",
    "onePiece_fotos",
    "SNKRS",
    "StarWars",
]
NAMES = [
    "AoT (",
    "deathNote (",
    "Evangel (",
    "KNY (",
    "LOTR (",
    "nba (",
    "OnePiece (",
    "Snkrs (",
    "SW (",
]


def extract_features(img, name, folder):
    if img is None:
        raise ValueError(f"Could not read image for {folder}/{name}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = cv2.mean(gray)[0]
    std_dev = cv2.meanStdDev(gray)[1][0][0]
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    n_edges = cv2.countNonZero(cv2.Canny(gray, 50, 150))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = (float(v) for v in cv2.mean(hsv)[:3])
    hist_r = [float(v) for v in cv2.calcHist([img], [2], None, [256], [0, 256])]
    hist_g = [float(v) for v in cv2.calcHist([img], [1], None, [256], [0, 256])]
    hist_b = [float(v) for v in cv2.calcHist([img], [0], None, [256], [0, 256])]

    return [name, folder, mean, std_dev, min_val, max_val, n_edges, hue, sat, val, hist_r, hist_g, hist_b]


def write_red_histogram_csv(destination=ROOT / "featuresHistR.csv"):
    with open(destination, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "Folder", "RedHistogram"])
        for folder, prefix in zip(FOLDERS, NAMES):
            for index in range(1, 101):
                name = f"{prefix}{index}).png"
                path = PHOTOS / folder / name
                img = cv2.imread(str(path))
                features = extract_features(img, name, folder)
                writer.writerow([features[0], features[1], *features[-3]])
    print(f"Wrote {destination}")


def explore():
    print("Folders:")
    for index, folder in enumerate(FOLDERS, start=1):
        print(f"{index}) {folder}")

    folder_index = int(input("Select a folder number: "))
    image_id = input("Choose an image from 1 to 100: ")
    name = f"{NAMES[folder_index - 1]}{image_id}).png"
    path = PHOTOS / FOLDERS[folder_index - 1] / name
    img = cv2.imread(str(path))
    features = extract_features(img, name, FOLDERS[folder_index - 1])
    print("Values", features[:10])

    colors = ("r", "g", "b")
    for histogram, color in zip(features[-3:], colors):
        plt.figure()
        plt.plot(histogram, color=color)
        plt.xlim([0, 256])
        plt.title(f"{name} — {color.upper()} histogram")
        plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        write_red_histogram_csv()
    else:
        explore()
