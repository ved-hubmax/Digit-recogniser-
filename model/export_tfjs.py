"""Export trained Keras dense weights in a small browser-friendly JSON format."""
import json
from pathlib import Path
import tensorflow as tf


root = Path(__file__).parent.parent
model_path = root / "model" / "mnist_dense_16_16.h5"
model = tf.keras.models.load_model(model_path)

hidden_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
size = [int(model.input_shape[-1]), *[layer.units for layer in hidden_layers]]
weights = []
biases = []
for layer in hidden_layers:
    kernel, bias = layer.get_weights()
    weights.append(kernel.T.tolist())
    biases.append(bias.tolist())

payload = {"sizes": size, "weights": weights, "biases": biases}
output = root / "web" / "model_weights.json"
output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
print("Saved", output)
print("Network sizes:", size)