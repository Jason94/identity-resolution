# Identity Resolution Transformer (IDRT)

IDRT is an open-source library designed to identify duplicate entries within structured contact data. IDRT uses a combination of traditional search algorithms and deep learning to accurately and quickly identify duplicate contact records at scale.

## Table of Contents

1. [Background](#background)
   * [Traditional Methods](#traditional-methods)
   * [IDRT's Solution - Deep Learning](#idrts-solution---deep-learning)

## Background

_Identity resolution_ is the process of identifying duplicate contacts in a database of people. This problem exists across a wide range of domains, such as customer, volunteer, or voter data. It's critical to keep your data clean as the people in your database move, change phone numbers and names, or simply fill out your forms with a typo.

### Traditional Methods - Fuzzy Searching and Machine Learning

The traditional approaches to identity resolution have largely revolved around fuzzy string comparisons and non-deep learning machine learning techniques. 

#### Fuzzy String Comparisons

Fuzzy string comparison is a technique used to calculate a "distance" between strings. The concept allows for less-than-perfect matches, which can help in detecting duplicates where the data entries may not be identical but are close. The degree of match is usually measured in terms of the Levenshtein distance, also known as the "edit distance", which calculates the minimum number of edits (insertions, deletions, substitutions) needed to transform one string into another. 

For example, fuzzy string comparisons would consider "John Smith" and "Jon Smyth" as a potential match despite the spelling differences. This method can be highly effective when the errors in data are due to human error, like typos or phonetic spelling. However, this approach might still struggle with more complex discrepancies, such as different aliases or extensive data inconsistencies. A fuzzy string comparison would not be able to identify that "Roberta Gonzales" and "Gonzales Roberta" are likely the same person, with the first and last names having accidentally been entered backwards.

#### Non-Deep Learning Machine Learning Approaches

Traditional machine learning approaches, which include methods like decision trees, logistic regression, and support vector machines, have also been employed for identity resolution. These methods typically involve crafting hand-engineered features based on domain knowledge, such as the similarity of names, addresses, or other contact information, and then training a model to predict whether two contacts are the same person based on these features. 

For example, one might create features such as the edit distance between two names, the geographical distance between two addresses, or whether two phone numbers are identical, and feed these features into a classifier to predict whether two entries refer to the same person.

While these methods can often yield decent results, they are limited by the quality and completeness of the hand-engineered features and the inherent limitations of the models used. For instance, it can be challenging to manually create features that capture all possible variations and nuances in the data.

### IDRT's Solution - Deep Learning

IDRT uses a technique known as _deep learning_ to identify duplicates in contact data. Unlike traditional ML models, IDRT's models are neural networks that are composed of millions of learnable parameters. In deep learning you usually do not manually specify features or rules. The complex models are able to learn the rules and patterns by training on hundreds of thousands of rows of training data.

For example, IDRT models have learned patterns like:
* If the first and last names between contacts are reversed, they can be treated similarly to the way they could be treated if the names were the same.
* Any text in an email after the `@` sign can be entirely ignored when evaluating a potential match.
* Common nicknames can be treated the same, like "Beth - Elizabeth" and "Bob - Robert".

All of these patterns were learned by the model during training without any direction from a human. This is the main advantage of deep learning and IDRT: the ability to learn extremely complex relationship between any number of fields of contact data.

The main limitation of deep learning 