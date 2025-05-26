import argparse
import logging
import random
from pathlib import Path
from typing import Optional

from dataset import Dataset
from model import Model


def tune(max_n: int, seed: Optional[int]) -> None:
    """Tune the hyperparameter of an n-gram model."""
    logging.basicConfig(level=logging.INFO)

    if seed is not None:
        random.seed(seed)

    dataset_filepath = Path(__file__).parent / "tiny_shakespeare.txt"
    dataset = Dataset(dataset_filepath)

    train_fraction = 0.99
    data_train, data_test = dataset.split(train_fraction)

    print(f"Dataset: train_size = {len(data_train)}, test_size = {len(data_test)}")

    models_dirpath = Path(__file__).parent / "models"
    models_dirpath.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Tuning hyperparameters with max_n={max_n}...")
    best_n = Model.hp_tune(data_train, data_test, max_n=max_n)

    print(f"Best model found with n={best_n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-n",
        type=int,
        default=7,
        help="Maximum n-gram size for hyperparameter tuning (default: 6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    max_n: int = getattr(args, "max_n")
    seed: Optional[int] = getattr(args, "seed")

    tune(max_n, seed)
