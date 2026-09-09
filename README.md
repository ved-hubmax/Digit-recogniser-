# Digit Recognizer Lab

A browser-based, MNIST-inspired handwritten digit recognizer. Draw a digit and inspect the signal through an interpretable dense network: **784 → 16 → 16 → 10**.

## Run the browser app

```powershell
python main.py
```

Open http://localhost:8000/web/ in your browser. The UI includes canvas capture, crop/resize/center-of-mass preprocessing, confidence bars, and a live weighted network diagram. The checked-in `web/model_weights.json` is a real trained model, not a demo classifier.

## Train real MNIST weights

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python model/train.py
python model/export_tfjs.py
```

Training currently reaches about 95.6% test accuracy with the intentionally small 784-16-16-10 network. `model/export_tfjs.py` exports the learned matrices directly to `web/model_weights.json`; this avoids a TensorFlow/TensorFlow.js converter compatibility problem and keeps inference client-side with no backend.

## Preprocessing

The canvas pipeline finds the non-empty bounding box, scales the longest edge into a 22×22 region, embeds it in a 28×28 image, aligns the center of mass to the image center, and flattens normalized grayscale pixels into the model's 784-value input.

## Project map

- `web/`: static application, drawing canvas, preprocessing, inference adapter, and visualization
- `model/train.py`: Keras training script
- `model/export_tfjs.py`: direct browser weight export script
- `web/model_weights.json`: trained weights used by the browser
- `main.py`: zero-dependency local server