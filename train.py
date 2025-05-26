import argparse
from pathlib import Path

from dataset import Dataset
from model import Model


def train(n: int) -> None:
    """Train a new n-gram model given a value of n."""
    dataset_filepath = Path(__file__).parent / "tiny_shakespeare.txt"
    dataset = Dataset(dataset_filepath)

    train_fraction = 0.99
    data_train, data_test = dataset.split(train_fraction)

    print(f"Dataset: train_size = {len(data_train)}, test_size = {len(data_test)}")

    models_dirpath = Path(__file__).parent / "models"
    models_dirpath.mkdir(parents=True, exist_ok=True)

    model = Model(n)

    print()
    print(f"Training model with n={n}...")
    model.train(data_train)

    model_filename = f"model-n{n}.pkl"
    model_filepath = models_dirpath / model_filename
    model.save(model_filepath)
    print(f"Model saved to ./models/{model_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        type=int,
        default=4,
        help="N-gram size for the model (default: 4)",
    )
    args = parser.parse_args()
    n: int = getattr(args, "n")

    train(n)
