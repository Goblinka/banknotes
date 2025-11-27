import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from src.utils import load_data


def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data("data/banknotes.csv")

    # Load model
    model_path = os.path.join("models", "banknote_model.keras")
    model = tf.keras.models.load_model(model_path)

    # Predict probabilities
    y_prob = model.predict(X_test)
    # Convert to 0/1 using threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int).ravel()

    # Metrics
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
