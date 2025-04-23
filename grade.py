import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from dataset import Dataset
from grading_utils import Case, Grader
from model import Model


class Case1a(Case):
    problem_name = "p1"
    case_name = "a"
    description = "Splitting a dataset into train and test sets"
    points = 2
    extra_credit = False

    def __call__(self) -> None:
        filepath = Path(__file__).parent / "tiny_shakespeare.txt"
        dataset = Dataset(filepath)

        train_fraction = 0.99
        train_size = dataset.get_train_size(train_fraction)
        expected = 1104240
        assert train_size == expected, "Expected train size does not match"
        assert isinstance(train_size, int), "Train size should be an integer"

        train_fraction = 0.90
        train_size = dataset.get_train_size(train_fraction)
        expected = 1003854
        assert train_size == expected, "Expected train size does not match"
        assert isinstance(train_size, int), "Train size should be an integer"


class Case2a(Case):
    problem_name = "p2"
    case_name = "a"
    description = "Normalizing a probability distribution"
    points = 3
    extra_credit = False

    def __call__(self) -> None:
        model = Model(4)
        prefix_counts = {"e": 13, "a": 8, "m": 4}
        distribution = model.normalize(prefix_counts)
        expected = {"e": 0.52, "a": 0.32, "m": 0.16}
        for key, value in distribution.items():
            assert self.is_close(value, expected[key], tol=0.001), (
                "Normalized distribution is incorrect"
            )

        model = Model(4)
        prefix_counts = {"m": 21, "o": 9, "t": 2}
        distribution = model.normalize(prefix_counts)
        expected = {"m": 0.65625, "o": 0.28125, "t": 0.0625}
        for key, value in distribution.items():
            assert abs(value - expected[key]) < 0.001, (
                "Normalized distribution is incorrect"
            )


class Case2b(Case):
    problem_name = "p2"
    case_name = "b"
    description = "Training a model"
    points = 2
    extra_credit = False

    def __call__(self) -> None:
        model = Model(1)
        model.train("Once upon a midnight dreary, while I pondered, weak and weary")
        assert len(model.table) == 22, "Model not trained correctly"
        assert len(model.table["n"]) == 4, "Model not trained correctly"
        assert model.table["n"]["c"] == 0.2, "Model not trained correctly"
        assert model.table["n"]["d"] == 0.4, "Model not trained correctly"


class Case3a(Case):
    problem_name = "p3"
    case_name = "a"
    description = "Sampling from a probability distribution"
    points = 3
    extra_credit = False

    def __call__(self) -> None:
        model = Model(2)
        k = 10000
        random.seed(42)
        counts: defaultdict[str, int]

        suffixes = ["a", "b", "c"]
        weights = [0.5, 0.3, 0.2]
        counts = defaultdict(int)
        for _ in range(k):
            sample = model.sample(suffixes, weights)
            counts[sample] += 1

        total = sum(counts.values())
        for suffix in suffixes:
            rate = float(counts[suffix]) / total
            expected = weights[suffixes.index(suffix)]
            assert self.is_close(rate, expected, tol=0.01), (
                f"Sampled suffix '{suffix}' does not match expected rate"
            )

        suffixes = ["a", "b", "c", "d"]
        weights = [0.35, 0.25, 0.25, 0.15]
        counts = defaultdict(int)
        for _ in range(k):
            sample = model.sample(suffixes, weights)
            counts[sample] += 1

        total = sum(counts.values())
        for suffix in suffixes:
            rate = float(counts[suffix]) / total
            expected = weights[suffixes.index(suffix)]
            assert self.is_close(rate, expected, tol=0.01), (
                f"Sampled suffix '{suffix}' does not match expected rate"
            )


class Case3b(Case):
    problem_name = "p3"
    case_name = "b"
    description = "Generating from a model"
    points = 2
    extra_credit = False

    def __call__(self) -> None:
        model = Model(2)
        model.train("Once upon a midnight dreary, while I pondered, weak and weary")
        random.seed(42)

        length = 10
        generated = model.generate(length, "On")
        assert len(generated) == length, "Generated text is not the correct length"
        expected = "ce upon a "
        assert generated == expected, "Generated text does not match expected output"

        length = 20
        generated = model.generate(length, "wh")
        assert len(generated) == length, "Generated text is not the correct length"
        expected = "ile I pon andereary,"
        assert generated == expected, "Generated text does not match expected output"


class Case4a(Case):
    problem_name = "p4"
    case_name = "a"
    description = "Tuning hyperparameters"
    points = 2
    extra_credit = True

    def __call__(self) -> None:
        random.seed(42)
        max_n = 4

        data_train = """
Once upon a midnight dreary, while I pondered, weak and weary,
Over many a quaint and curious volume of forgotten lore,
While I nodded, nearly napping, suddenly there came a tapping,
As of some one gently rapping, rapping at my chamber door. “
“'Tis some visitor,” I muttered, “tapping at my chamber door—
Only this, and nothing more.”
Ah, distinctly I remember it was in the bleak December,
And each separate dying ember wrought its ghost upon the floor.
Eagerly I wished the morrow;—vainly I had sought to borrow
From my books surcease of sorrow—sorrow for the lost Lenore—
For the rare and radiant maiden whom the angels name Lenore—
Nameless here for evermore."""
        data_test = """
Deep into that darkness peering, long I stood there wondering, fearing,
Doubting, dreaming dreams no mortals ever dared to dream before;
But the silence was unbroken, and the stillness gave no token,
And the only word there spoken was the whispered word, “Lenore!”
This I whispered, and an echo murmured back the word, “Lenore!”—
Merely this, and nothing more."""
        best_n = Model.hp_tune(data_train, data_test, max_n=max_n)
        expected = 2
        assert best_n == expected, (
            "Got incorrect value of hyperparameter n after tuning"
        )

        filepath = Path(__file__).parent / "tiny_shakespeare.txt"
        dataset = Dataset(filepath)
        train_fraction = 0.99
        data_train, data_test = dataset.split(train_fraction)
        best_n = Model.hp_tune(data_train, data_test, max_n=max_n)
        expected = 4
        assert best_n == expected, (
            "Got incorrect value of hyperparameter n after tuning"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_name", nargs="?", type=str, default=None)
    args = parser.parse_args()
    problem_name: Optional[str] = getattr(args, "problem_name")

    cases: list[Case] = [
        Case1a(),
        Case2a(),
        Case2b(),
        Case3a(),
        Case3b(),
        Case4a(),
    ]

    grader = Grader(cases=cases)
    grader.run(problem_name)


if __name__ == "__main__":
    main()
