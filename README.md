# Digit Recognizer Lab

A browser-based, MNIST-inspired handwritten digit recognizer. Sketch a digit on the canvas and watch the signal propagate through an interpretable dense neural network in real time: **784 → 16 → 16 → 10**.

---

## Features

- **Real-Time Client-Side Inference**: Runs fully in the browser via native JavaScript vector math (`web/model_loader.js`) with zero external runtime dependencies.
- **Accurate MNIST Preprocessing**: Implements standard MNIST image normalization (luminance thresholding, 20×20 bounding box aspect-ratio preservation, anti-aliasing downsampling, and center-of-mass centering).
- **Interactive Network Visualization**: Real-time rendering of all network layers, neuron activations, and weighted synaptical connections on an HTML5 canvas.
- **Confidence Readout & Probabilities**: Live readout showing top prediction and softmax confidence bars for all digits (0–9).
- **Zero-Backend Deployment**: Fully static and deployable to GitHub Pages.

---

## Run the Browser App Locally

Start the local development server:

```powershell
python main.py
```

Then open **http://127.0.0.1:8000/web/** in your browser.

---

## Live GitHub Pages App

- **Repository**: [https://github.com/ved-hubmax/Digit-recogniser-](https://github.com/ved-hubmax/Digit-recogniser-)
- **Live Demo**: [https://ved-hubmax.github.io/Digit-recogniser-/](https://ved-hubmax.github.io/Digit-recogniser-/)

The application runs purely client-side as a static GitHub Pages site without requiring a backend server.

### Deploying to GitHub Pages

1. Go to your repository on GitHub.
2. Navigate to **Settings > Pages**.
3. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
4. Select branch `main` and folder `/ (root)`.
5. Click **Save**. The root `index.html` automatically redirects visitors to the `web/` application.

---

## Preprocessing Pipeline

The canvas preprocessing pipeline matches the official MNIST normalization standards to ensure high prediction accuracy on real human handwriting:

1. **Ink Detection**: Detects stroke boundaries using luminance thresholding on the dark canvas.
2. **Aspect-Ratio Preserved Scaling**: Crops the bounding box and scales it into a 20×20 box inside a 28×28 grid.
3. **Anti-Aliasing**: Downsamples strokes with high-quality smoothing to avoid broken or jagged lines.
4. **Grayscale Normalization**: Normalizes pixel values from `0.0` (empty background) to `1.0` (digit strokes).
5. **Center-of-Mass Centering**: Computes the true center of mass of the stroke pixels and translates the digit to the center `(13.5, 13.5)` with edge clamping.

---

## Training & Model Export

To retrain the neural network or export custom weights:

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model on MNIST
python model/train.py

# 4. Export weights to browser JSON format
python model/export_tfjs.py
```

- `model/train.py` trains the dense neural network on MNIST using Keras and saves the best model weights to `model/mnist_dense_16_16.h5` (~95.5%+ test accuracy).
- `model/export_tfjs.py` extracts layer weights and biases into `web/model_weights.json`. This lightweight format loads directly into the browser without TensorFlow.js dependency overhead.

---

## Project Structure

```
Digit recogniser/
├── index.html            # Root redirect for GitHub Pages
├── main.py               # Lightweight local Python server
├── requirements.txt      # Python dependencies (TensorFlow, Pillow, etc.)
├── README.md             # Project documentation
├── model/
│   ├── train.py          # Keras model definition and training
│   ├── export_tfjs.py    # Exports H5 model weights to JSON
│   └── mnist_dense_16_16.h5 # Trained Keras model
└── web/
    ├── index.html        # Main web interface
    ├── styles.css        # UI styling and layout
    ├── canvas.js         # Drawing interaction and MNIST preprocessing
    ├── model_loader.js   # Browser neural network inference engine
    ├── network_viz.js    # Interactive neural network visualization
    └── model_weights.json# Serialized neural network weights & biases
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
