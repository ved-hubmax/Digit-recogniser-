"""Train the interpretable 784-16-16-10 MNIST model and save it as H5."""
from pathlib import Path
import tensorflow as tf


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,), name="pixels"),
        tf.keras.layers.Dense(16, activation="relu", name="hidden_1"),
        tf.keras.layers.Dense(16, activation="relu", name="hidden_2"),
        tf.keras.layers.Dense(10, activation="softmax", name="digits"),
    ])


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train.reshape(-1, 784) / 255.0, x_test.reshape(-1, 784) / 255.0
    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_split=.1, epochs=25, batch_size=128)
    print("Test accuracy:", model.evaluate(x_test, y_test, verbose=0)[1])
    output = Path(__file__).parent / "mnist_dense_16_16.h5"
    model.save(output)
    print("Saved", output)