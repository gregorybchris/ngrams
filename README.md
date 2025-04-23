# N-grams

Introduction to generative language modeling using an n-gram model.

This project is an assignment for the Park Tudor data science class. See [assignment.md](./assignment.md) for detailed instructions.

## Files

| Name                                           | Description                                     |
| ---------------------------------------------- | ----------------------------------------------- |
| [assignment.md](./assignment.md)               | The instructions for the assignment             |
| [tiny_shakespeare.txt](./tiny_shakespeare.txt) | The dataset we use to train our language model  |
| --                                             | --                                              |
| [dataset.py](./dataset.py)                     | Utilities for loading and splitting the dataset |
| [model.py](./model.py)                         | The n-gram model implementation                 |
| --                                             | --                                              |
| [train.py](./train.py)                         | A CLI script to train the model                 |
| [generate.py](./generate.py)                   | A CLI script to generate text with the model    |
| [grade.py](./grade.py)                         | A CLI script to grade the assignment            |
| --                                             | --                                              |
| [grading_utils.py](./grading_utils.py)         | Utilities for grading, can be ignored           |

## Dataset

The [Tiny Shakespeare dataset](./tiny_shakespeare.txt) has been downloaded from [the GitHub of Andrej Karpathy](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt).
