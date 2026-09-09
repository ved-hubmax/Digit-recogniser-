"""Export trained Keras weights in a small browser-friendly JSON format."""
import json
from pathlib import Path
import tensorflow as tf


root = Path(__file__).parent.parent
model = tf.keras.models.load_model(root / "model" / "mnist_dense_16_16.h5")
weights = model.get_weights()
payload = {
    "sizes": [784, 16, 16, 10],
    "weights": [weights[0].T.tolist(), weights[2].T.tolist(), weights[4].T.tolist()],
    "biases": [weights[1].tolist(), weights[3].tolist(), weights[5].tolist()],
}
output = root / "web" / "model_weights.json"
output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
print("Saved", output)