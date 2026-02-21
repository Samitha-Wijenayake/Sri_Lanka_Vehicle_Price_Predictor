import json
import os
import pickle

def save_json(data, filepath):
    """Saves a dictionary to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved JSON data to {filepath}")

def load_json(filepath):
    """Loads a dictionary from a JSON file."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return {}
    with open(filepath, "r") as f:
        return json.load(f)

def save_pickle(model, filepath):
    """Saves a python object (model/encoder) using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_pickle(filepath):
    """Loads a pickeled python object."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    with open(filepath, "rb") as f:
        return pickle.load(f)
