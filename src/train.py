import os
import tensorflow as tf

from src.utils import load_data


def build_model(input_dim: int = 4) -> tf.keras.Model:
    """
    Builds a simple feedforward neural network for binary classification.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(8, input_shape=(input_dim,), activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main():
    # 1) Load data
    X_train, X_test, y_train, y_test = load_data("data/banknotes.csv")

    # 2) Build model
    model = build_model(input_dim=X_train.shape[1])

    # 3) Train
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,  # deo train-a za validaciju
        verbose=1,
    )

    # 4) Evaluate on test set
    results = model.evaluate(X_test, y_test, verbose=0)
    print("Test results:")
    for name, value in zip(model.metrics_names, results):
        print(f"  {name}: {value:.4f}")

    # 5) Save model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "banknote_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
