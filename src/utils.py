import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str, test_size: float = 0.4, random_state: int = 42):
    """
    Loads the banknote dataset and returns train/test splits.
    """
    data = pd.read_csv(path)

    # First 4 columns = features (evidence)
    evidence = data.iloc[:, :4].values

    # Last column = label (0/1 in CSV) -> invert if desired
    labels = data.iloc[:, 4].apply(lambda x: 1 if x == 0 else 0).values

    X_train, X_test, y_train, y_test = train_test_split(
        evidence,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,  # da zadržiš odnos klasa
    )
    return X_train, X_test, y_train, y_test
