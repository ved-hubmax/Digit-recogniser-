"""Train a stronger MNIST model and save it as H5."""
from pathlib import Path
import tensorflow as tf


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,), name="pixels"),
        tf.keras.layers.Dense(128, activation="relu", name="hidden_1"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu", name="hidden_2"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation="softmax", name="digits"),
    ])


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=40,
        batch_size=128,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    output = Path(__file__).parent / "mnist_dense_128_64.h5"
    model.save(output)
    print("Saved", output)
    print("Best validation accuracy:", max(history.history["val_accuracy"]))