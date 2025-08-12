import pandas as pd

def load_csv(path: str):
    """Load email CSV with columns: id, text, label"""
    df = pd.read_csv(path)
    return df
