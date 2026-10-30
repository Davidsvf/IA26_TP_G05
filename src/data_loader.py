
from pathlib import Path
import pandas as pd


def load_data(dataset_path):
    """Lê todos os CSV de uma pasta e devolve um dicionário."""
    folder = Path(dataset_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {folder}")

    data = {}
    for file in sorted(folder.glob("*.csv")):
        data[file.stem] = pd.read_csv(file)

    if not data:
        raise ValueError(f"Não existem ficheiros CSV em {folder}")
    return data
