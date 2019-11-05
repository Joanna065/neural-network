import os
import pickle


def load_data(filename):
    with open(filename, "rb") as f:
        train_data, val_data, test_data = pickle.load(f, encoding="latin1")
    return train_data, val_data, test_data


def ensure_dir_path_exists(path):
    """Checks if path is an existing directory and if not, creates it."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
