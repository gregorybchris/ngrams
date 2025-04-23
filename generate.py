import argparse
import random
from pathlib import Path
from typing import Optional

from model import Model


def generate(n: int, length: int, prefix: str, seed: Optional[int]) -> None:
    """Generate text using a trained model."""
    if seed is not None:
        random.seed(seed)

    if len(prefix) != n and prefix != "":
        raise ValueError(f"Prefix length must be {n} if a prefix is provided")

    models_dirpath = Path(__file__).parent / "models"
    if not models_dirpath.exists():
        raise ValueError("Models directory does not exist, please run train.py first")

    model_filepath = models_dirpath / f"model-n{n}.pkl"
    if not model_filepath.exists():
        raise ValueError(
            f"Model file {model_filepath} does not exist, please run train.py -n {n} first"
        )

    model = Model.load(model_filepath)

    prefix = prefix if prefix is not None else ""

    generated = model.generate(length, prefix)
    print(generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        type=int,
        help="N-gram size",
        default=4,
    )
    parser.add_argument(
        "--length",
        type=int,
        default=300,
        help="Length of the generated text",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to start the generation with",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    args = parser.parse_args()
    n: int = getattr(args, "n")
    length: int = getattr(args, "length")
    prefix: str = getattr(args, "prefix")
    seed: Optional[int] = getattr(args, "seed")

    generate(n, length, prefix, seed)
