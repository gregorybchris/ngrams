from pathlib import Path


class Dataset:
    data: str

    def __init__(self, filepath: Path):
        with filepath.open("r") as fp:
            self.data = fp.read()

    def get_train_size(self, train_fraction: float) -> int:
        """Get the number of characters in the training set based on the training fraction and full dataset size."""
        # - - - Problem 1 - - -
        # Implement your code here, your answer should be 1-3 lines of Python.
        # To get started, consider using the len() function.
        # Keep in mind that you can cast a float to an int with the int() function.
        # train_fraction is a float between 0 and 1 (exclusive).
        # * * Implementation starts here * * *

        # * * Implementation ends here * * * *
        raise NotImplementedError

    def split(self, train_fraction: float) -> tuple[str, str]:
        """Split the dataset into training and test sets."""
        train_size = self.get_train_size(train_fraction)
        data_train = self.data[:train_size]
        data_test = self.data[train_size:]
        return data_train, data_test
