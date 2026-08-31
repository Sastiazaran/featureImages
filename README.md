# Frame Lab

Upload a still. The app reads **color, edges, and RGB histograms**, then classifies the image into one of nine visual worlds with a one-vs-rest logistic regression model.

| Class | World |
| --- | --- |
| 1 | Attack on Titan |
| 2 | Death Note |
| 3 | Evangelion |
| 4 | Demon Slayer (Kimetsu no Yaiba) |
| 5 | The Lord of the Rings |
| 6 | NBA |
| 7 | One Piece |
| 8 | Sneakers (SNKRS) |
| 9 | Star Wars |

This started as a Graphic Simulation class project: extract features from a labeled set of 900 frames, train logistic regression, then classify a new image from the browser.

## How it works

1. The browser drop zone sends the file to `POST /upload`.
2. The Express server saves the image under `myapp/server/fotos/`.
3. `classify.py` builds the same **776-d** vector used in training:
   - grayscale mean, standard deviation, min, max
   - Canny edge count
   - HSV channel means
   - 256-bin histograms for R, G, and B
4. Saved weights in `myapp/server/data/theta_List.csv` score all nine classes. The highest logit wins.

## Run it

```bash
python3 -m pip install -r requirements.txt
cd myapp/server
npm install
cd ..
npm start
```

Open [http://localhost:3000](http://localhost:3000), drop a JPEG/PNG/GIF/WebP still, and read the lab report.

Classify a file from the terminal:

```bash
python3 myapp/server/classify.py "imageClassifier/fotosGabriel/attackOnTitan_fotos/AoT (1).png"
```

## Tests

```bash
python3 myapp/server/test_classify.py
```

The suite checks that live OpenCV features match the training CSV, that a known Attack on Titan still is labeled correctly, and that argmax accuracy on the 900 training rows stays above 75%.

## Project layout

```
index.html / dropArea.css / imageDrop.js   UI (also served from myapp/)
myapp/server/app.js                        Express API + static files
myapp/server/classify.py                   Inference
myapp/server/imageClassifier.py            Training math (logistic regression)
myapp/server/data/theta_List.csv           Trained one-vs-rest weights
imageClassifier/fotosGabriel/              Labeled stills used to train
imageClassifier/main.py                    Feature extraction helper
```

## Training data helper

`imageClassifier/main.py` can inspect a still from the labeled folders. Pass `extract` only if you intend to rebuild `featuresHistR.csv`:

```bash
python3 imageClassifier/main.py
python3 imageClassifier/main.py extract
```
