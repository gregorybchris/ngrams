import logging
import pickle
import random
from pathlib import Path
from typing import Self

logger = logging.getLogger(__name__)


class Model:
    """
    Example of a model trained with n=3 (3-grams):

    In this example, the strings "you" and "I h" are prefixes. For the prefix "you", the characters
    " ", "r", and "n" are suffixes. The probabilities indicate how likely each suffix is to follow the
    prefix. For example, the space character is 50% likely to follow "you", while "r" is 40% likely
    and "t" is 10% likely.

    table = {
        "you": {
            " ": 0.5,
            "r": 0.4,
            "t": 0.1,
        },
        "I h": {
            "a": 0.8,
            "e": 0.1,
            "i": 0.05,
            "o": 0.05,
        },
    }

    Note that the probabilities here are "normalized" -- They sum to 1.0.
    """

    table: dict[str, dict[str, float]]
    n: int

    def __init__(self, n: int):
        self.n = n
        self.table = {}

    def train(self, data: str) -> None:
        """Train an n-gram model on the given data."""
        # Count up the occurrences of each prefix and suffix
        counts: dict[str, dict[str, int]] = {}
        for i in range(self.n, len(data)):
            prefix = data[i - self.n : i]
            suffix = data[i]
            if prefix not in counts:
                counts[prefix] = {}
            if suffix not in counts[prefix]:
                counts[prefix][suffix] = 0
            counts[prefix][suffix] += 1

        # Normalize the counts to probabilities
        # For each prefix the sum of suffix probabilities should be 1.0
        for prefix, prefix_counts in counts.items():
            self.table[prefix] = self.normalize(prefix_counts)

    def normalize(self, prefix_counts: dict[str, int]) -> dict[str, float]:
        """Convert a dict from prefix->count to a dict from prefix->probability."""
        # - - - Problem 2 - - -
        # Implement your code here, your answer should be 3-5 lines of Python.
        # Here we want to normalize the counts to probabilities.
        # The final dict should have probabilities that sum to 1.
        # Hint: You can use the sum() function to get the total count.
        # Hint: You can use items() to iterate over key, value pairs in a dictionary.
        # Hint: You can use values() to iterate over values of a dictionary.
        # * * Implementation starts here * * *

        # * * Implementation ends here * * * *
        raise NotImplementedError

    def generate(self, length: int, prefix: str) -> str:
        """Generate a string of the given length using the model."""
        generated = ""
        for _ in range(length):
            # Handle the edge case that our prefix is not in the training set
            if prefix not in self.table:
                rand = random.randint(0, len(self.table) - 1)
                prefix = list(self.table.keys())[rand]
                distribution = self.table[prefix]
            else:
                distribution = self.table[prefix]

            # Sample a suffix from the prefix distribution
            suffixes = list(distribution.keys())
            weights = list(distribution.values())
            suffix = self.sample(suffixes, weights)

            generated += suffix
            prefix = prefix[1:] + suffix
        return generated

    def sample(self, suffixes: list[str], weights: list[float]) -> str:
        """Sample a single character from the prefix distribution."""
        # - - - Problem 3 - - -
        # Implement your code here, your answer should be 1-3 lines of Python.
        # Here we want to sample a single character from the prefix distribution.
        # Hint: random.choices() is a function that takes a list of items to sample from,
        # a list of weights, and a value k that indicates how many items to sample.
        # It returns a list of the sampled items, but if k=1, you can just take the first element.
        # https://docs.python.org/3/library/random.html#random.choices
        # * * Implementation starts here * * *

        # * * Implementation ends here * * * *
        raise NotImplementedError

    def evaluate(self, data: str) -> float:
        """Evaluate the model's accuracy on the given data."""
        n_correct = 0
        for i in range(self.n, len(data)):
            prefix = data[i - self.n : i]
            suffix_predicted = self.generate(1, prefix)
            suffix_actual = data[i]
            if suffix_predicted == suffix_actual:
                n_correct += 1

        total = len(data) - self.n
        accuracy = n_correct / total
        logger.info(f"n={self.n}, accuracy={accuracy:.4f}")
        return accuracy

    @classmethod
    def hp_tune(cls, data_train: str, data_test: str, *, max_n: int) -> int:
        """Train a family of models with different n-gram sizes and return the best value of n."""
        # - - - Problem 4 (extra credit) - - -
        # Implement your code here, your answer should be 8-15 lines of Python.
        # In this problem we want to tune the hyperparameters of our model.
        # We just have a single hyperparameter n, which is the size of the n-grams.
        # Our hp_tune() method should do a sweep over possible values of n
        # to find the value of n that maximizes accuracy.
        # Hint: You have access to the train() and evaluate() methods. You should
        # use them here to make your implementation simpler.
        # Hint: To create a new instance of the Model class in this classmethod
        # you can use cls(n) instead of Model(n).
        # Hint: Be careful with the bounds of your range(). The smallest n we should use is 1.
        # * * Implementation starts here * * *

        # * * Implementation ends here * * * *
        raise NotImplementedError

    def save(self, filepath: Path) -> None:
        """Save the model to a file."""
        with filepath.open("wb") as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, filepath: Path) -> Self:
        """Load the model from a file."""
        with filepath.open("rb") as fp:
            return pickle.load(fp)
