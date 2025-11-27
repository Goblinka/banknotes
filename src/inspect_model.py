import os
import tensorflow as tf


def main():
    model_path = os.path.join("models", "banknote_model.keras")
    model = tf.keras.models.load_model(model_path)

    print("Model summary:")
    model.summary()

    # težine prvog sloja
    W1, b1, W2, b2 = model.get_weights()
    print("\nFirst layer weights shape:", W1.shape)
    print("First layer bias shape:", b1.shape)


if __name__ == "__main__":
    main()
