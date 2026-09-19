"""Train a strong MNIST dense model and save it as H5."""
from pathlib import Path
import tensorflow as tf


def build_model(hidden_units=(16, 16)):
    """Build a sequential dense neural network matching the browser architecture."""
    layers = [tf.keras.layers.Input(shape=(784,), name="pixels")]
    for i, units in enumerate(hidden_units, 1):
        layers.append(tf.keras.layers.Dense(units, activation="relu", name=f"hidden_{i}"))
        layers.append(tf.keras.layers.Dropout(0.1))
    layers.append(tf.keras.layers.Dense(10, activation="softmax", name="digits"))
    return tf.keras.Sequential(layers)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    hidden_units = (16, 16)
    model = build_model(hidden_units)
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
        epochs=35,
        batch_size=128,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")

    name = f"mnist_dense_{'_'.join(str(u) for u in hidden_units)}.h5"
    output = Path(__file__).parent / name
    model.save(output)
    print("Saved", output)
    print("Best validation accuracy:", max(history.history["val_accuracy"]))