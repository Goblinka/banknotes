# Banknote Classifier 🧠💸

A small neural network project that classifies banknotes as **authentic** or **counterfeit** using TensorFlow and a simple feedforward network.

This is a minimal end-to-end example of:
- loading tabular data with **pandas**
- splitting data into train / test with **scikit-learn**
- building and training a **Keras** model (TensorFlow)
- evaluating the model using **accuracy, precision, recall** and a **confusion matrix**

---

## 📊 Dataset

The project assumes a CSV file with **5 columns**:

- first 4 columns → numeric features (e.g. statistics extracted from banknote images)
- 5th column → binary label (`0` / `1`)

By default, the code:
- uses the first 4 columns as **features**
- inverts the last column so that:

```text
original label 0 → mapped to 1
original label 1 → mapped to 0

Place your CSV file here:
data/banknotes.csv

🧱 Model

Input:  4 features
Hidden: Dense(8), ReLU
Output: Dense(1), Sigmoid
Loss:   binary_crossentropy
Opt:    Adam
Metrics: accuracy, precision, recall

🚆 Training

python -m src.train

After training, the model is saved to:
models/banknote_model.keras
You can inspect the model architecture and view its weights using the helper script:
python -m src.inspect_model

✅ Evaluation

python -m src.evaluate

