# Assignment

## Overview

In AI and machine learning we have a concept of "auto-regressive" language modeling. It's a fancy word that just means we feed text to a model to predict the text that follows and then pass the output text back into the model as input, continuing this loop to generate longer and longer text.

For example, let's say you start with some text "I can't believe". You can use your model to predict the next word. Let's say the model predicts "you". Let's append that to our initial text to get "I can't believe you". We can feed this new text back into the same model. Perhaps the model predicts the word "would" next. Now the updated full text is "I can't believe you would". You can keep repeating this process and as long as our model is smart about how it selects the next word our text can be intelligible or ideally even useful.

In this assignment we're going to train our own auto-regressive language model! The specific type of model we'll be training is called an n-gram model and while it's relatively simple, it has a lot of similarities conceptually to ChatGPT and other cutting edge AI that you're likely familiar with.

For the sake of simplicity, instead of predicting the next word as shown above (or the next "token" as done by ChatGPT) we'll be predicting the next character.

## What is an n-gram?

An n-gram is a n-symbol sequence. In the context of this project, it's useful to think of an n-gram as an n-letter substring. For example in the word "Panther", the substring "Pan" is a 3-gram. Considering a prefix "Pan" we can take an educated guess at what the next letter might be. "Pan" could be a prefix to "Pants" or even "Pangolin".

An n-gram model looks at a substring of n letters in order to predict the next letter. In our example above, we might assign the character 't' a probability of 70% and the character 'g' a probability of 30% since 't' is the suffix of "Pan" in both "Panther" and "Pants" while 'g' only follows "Pan" in "Pangolin".

A 2-gram is commonly known as a "bigram" and a 3-gram is commonly known as a "trigram".

You can read more about n-gram language models [on Wikipedia here](https://en.wikipedia.org/wiki/Word_n-gram_language_model).

## Goals

By the end of this assignment you will have trained your own model that can generate text.

Other concepts you'll learn about:

- Dataset splitting
- Probability distributions
- Sampling

## Tips

- Pair up in groups of two students - the problems must be done in order, collaborate to avoid getting stuck
- Run the grader often - it can help you debug (`python grade.py`)
- Use lots of print statements - print out variables you don't understand
- Read the hints in the code multiple times - they are designed to keep you on track
- Trust the recommended number of lines - they're designed to help you not overcomplicate your solution
- All code should be written between lines with asterisks - you shouldn't touch anything else
- Problem 4 is extra credit

## Problem 1

Open `dataset.py` and implement the `get_train_size()` method.

In this problem you'll split your dataset into a training set and a test set. This allows you to train our model on some data, while keeping some of your data separate. This separate test set will be used later when you want to evaluate the quality of your model.

You can think of this like a quiz at the end of a unit in school. If you quiz someone on the same problems they saw during lessons it'll be too easy! It's better to keep some questions a secret until the quiz. In school we just call this learning. In AI we call this "generalizing" to unseen data.

Put your solution between the \* symbols. You should not edit and code outside the lines with \* symbols.

Run `python grade.py p1` to check your solution.

If your implementation of `get_train_size()` determines the correct value of the training set size you should see the following output in your terminal:

```rb
==========================================
• ✅  p1-a ==> [2/2] passed
• ⏳  p2-a ==> [0/3] unimplemented
• ⏳  p2-b ==> [0/2] unimplemented
• ⏳  p3-a ==> [0/3] unimplemented
• ⏳  p3-b ==> [0/2] unimplemented
• ⏳  p4-a ==> [0/0] unimplemented (extra credit)
==========================================
Score: 2/12
==========================================
```

## Problem 2

Open `model.py` and implement the `normalize()` method.

In this problem you'll calculate the probabilities for possible continuations of the text. For example, if your text starts with "I believ" you might expect the text to continue on to read "I believe".

This is used during training to determine which continuations are most likely. This is the meat of the model training process for n-gram models.

It may be helpful to read the docstring in `model.py` for the `Model` class. It shows how the model stores the probabilities of suffixes given all prefixes seen during training.

Run `python grade.py p2` to check your solution.

Once tests pass you can try running `python train.py` to train a model.

If training completes successfully you should see the following output in your terminal:

```rb
Dataset: train_size = 1104240, test_size = 11154

Training model with n=4...
Model saved to ./models/model-n4.pkl
```

You should also see a trained model file show up in the [models](./models) folder called [model-n4.pkl](./models/model-n4.pkl). We can't do anything with it quite yet, but we will in the next problem.

## Problem 3

Open `model.py` and implement the `sample()` method.

In this problem you'll use your trained model to start generating text. You should select the next character by using the probability distribution from problem 2. Characters that the model finds more likely should be sampled with higher frequency.

Make sure to pay close attention to the hints in the code. You should take advantage of functions available to you in Python's standard library.

Run `python grade.py p3` to check your solution.

Once tests pass you should run `python generate.py --seed 42 --prefix "ROME" --length 185` to generate some text.

```txt
ROMEO:
And not such a light I double Gloucestow appear'd of a dry:
May the know farewell further.

HORTENSIO:
I am big-swoln heard kill scorneys of them o' the citizens!

PRINCENTIO:
And no
```

You can play around with the `--length`, `--seed`, and `--prefix` options to change the length of text generated, the random seed, and the prefix text from which to start generating.

Now that you've got a handle on generation, let's play around with different values of `n`. You can run `python train.py -n 1` to train a different n-gram model then run `python generate.py -n 1` to test it. Test with different values of `n` and see what you notice about the quality of generated text.

## Problem 4 (extra credit)

Open `model.py` and implement the `hp_tune()` method.

In AI, a "hyperparameter" is a variable that alters the model's learning process. In contrast to a parameter, a hyperparameter is _not_ something the model learns during training. A hyperparameter is set before training starts and is hand-selected by the AI researcher.

In n-grams the value `n` is a hyperparameter. It configures the number of characters (or words) we look at to determine the next character.

So how do you choose `n`? One way is through "hyperparameter tuning" a process that searches through possible values of our hyperparameters and selects the one that produces the best model.

In this problem you'll iterate over possible values of `n` (from 1 to `max_n`), train a model with each value of `n`, evaluate the accuracy of each, and return the value of `n` that maximizes accuracy.

Run `python grade.py p4` to check your solution.

Now try running `python tune.py`. If your solution is correct you should see the following output in your terminal:

```rb
Dataset: train_size = 1104240, test_size = 11154

Tuning hyperparameters with max_n=7...
INFO:model:n=1, accuracy=0.1564
INFO:model:n=2, accuracy=0.2677
INFO:model:n=3, accuracy=0.3802
INFO:model:n=4, accuracy=0.4488
INFO:model:n=5, accuracy=0.4547
INFO:model:n=6, accuracy=0.4164
INFO:model:n=7, accuracy=0.3596
Best model found with n=5
```

Hyperparameter tuning can be very slow and resource intensive. The best AI researchers have very good intuitions about how to set hyperparameters without tuning and these intuitions help save massive time and resources when training cutting edge large language models.

## Food for thought

### Parameter counts

For our n-gram model, a parameter is a single floating point probability stored in the model's lookup table. Assuming the training set has only 27 unique characters including lowercase letters and spaces, in terms of n, what is the maximum number of parameters your model would have?

Hint: the number of possible permutations of 3 characters would be `27 * 27 * 27` or `27^3`.

### Model capacity

Model "capacity" is approximately the number of parameters the model has. Roughly it's how much _stuff_ you can fit into it during training. Does increasing the value n always increase model accuracy? Why or why not? What affect would increasing the training set size have on accuracy? Use the concept of model capacity in your answer.

### OOD

In machine learning we have a concept called "out of distribution" (or OOD) which means we're trying to predict based on input that is unlike anything the model looked at during training. What do you expect the behavior of the model to be when we generate text starting with an OOD prefix?

Hint: Reading the implementation of `Model.generate()` may be informative.

Hint: Imagine Mr. Ritz gives you a multiple choice quiz on a topic unlike anything he taught in class. What would your answers look like?

### Context length

As we've seen in problems 3 and 4, we can get higher quality generations by taking into consideration more history or "context". If we wanted to generate a whole novel that's logically consistent we'd benefit from letting the model consider a whole lot of context.

What sort of texts might require a lot of context? What sort of texts might be easy to model with a very short context? List an example of each and briefly explain your thinking.

### Inference speed

One downside of auto-regressive models is that you need to iteratively pass the output back through the model as input. The whole output text cannot just be generated at once. This loop can be very slow if you're generating a long text, especially as your model gets larger. Wouldn't it be nice if we could predict multiple characters at a time? (a suffix with length > 1). What do you expect to happen to accuracy as we increase our suffix length? Why? If we double our suffix length, how should we expect the generation speed to change? Why?

For a deeper dive consider reading about "multi-token prediction" or "speculative decoding". In image generation with diffusion models there are analogous methods, one of which is DDIM (denoising diffusion implicit models).
