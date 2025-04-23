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

    print(f"Training size: {len(data_train)}")
    print(f"Test size: {len(data_test)}")

    models_dirpath = Path(__file__).parent / "models"
    models_dirpath.mkdir(parents=True, exist_ok=True)

    model = Model(n)

    print()
    print(f"Training model with n={n}")
    model.train(data_train)

    print(f"Evaluating model with n={n}")
    accuracy = model.evaluate(data_test)
    print(f"Accuracy: {accuracy:.2}")

    model_filepath = models_dirpath / f"model-n{n}.pkl"
    model.save(model_filepath)


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
