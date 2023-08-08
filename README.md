# Identity Resolution Transformer (IDRT)

IDRT is an open-source library designed to identify duplicate entries within structured contact data. IDRT uses a combination of traditional search algorithms and deep learning to accurately and quickly identify duplicate contact records at scale. Particularly, IDRT uses a _transformer_ neural network, which is a kind of neural network that was discovered at Google in 2017 and powers some of the most complex AI's like Chat GPT.

## Table of Contents

1. [Overview](#overview)
1. [Background](#background)
   * [Traditional Methods](#traditional-methods)
   * [IDRT's Solution - Deep Learning](#idrts-solution---deep-learning)
1. [How IDRT Works](#how-idrt-works)
   * [Step 1: Generate vector encodings for each contact](#step-1-generate-vector-encodings-for-each-contact)

## Overview

Under the hood, IDRT uses a technique called _machine learning_ to identify patterns and learn to effectively classify duplicate & distinct contacts. Any program that utilizes machine learning is composed of two pieces: the computer code and the _model_, which is a set of numbers that tells the code how to mathematically weight different factors. The relationship between machine-learning software and a model is like the relationship between computer hardware and computer code: without the _model_ the machine-learning _software_ cannot actually do anything. This is similar to how a blank computer, without any code installed, cannot actually do anything.

(_TODO:_ Actually move all of the training code into a sub-package called "studio"...)

IDRT contains two main sub-packages: _IDRT.algorithm_ and _IDRT.studio_. _IDRT.algorithm_ contains a set of runnable scripts that use an IDRT model to perform an efficient duplicate search on a database of contacts. _IDRT.studio_ contains tools that allow you to train your own IDRT model based on existing duplicate/distinct data. Note that you can use an existing model to run the IDRT matching algorithm, without having to train your own!

This readme contains:
* A general overview of the project
* Background information about the deep learning technique and why it is suitable for identity resolution
* A conceptual outline of the matching algorithm contained in _IDRT.algorithm_
* A description of the structure of the underlying neural network

For more detailed documentation on how to run the matching algorithm, see the [Readme for the IDRT.algorithm sub-package](). For more detailed documentation on how to train your own IDRT model, see the [Readme for the IDRT.studio sub-package]().

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

Deep learning is primarily limited by the quantity and quality of available training data. We have had success training models with ~300,000 rows of example data. Training is also generally unviable on CPU hardware; you need access to a GPU (graphics card) to practically train a model. _(With that said, training on GPU hardware scales well. We were able to train a complete IDRT model pair in under three hours on a seven-year-old Nvidia GTX 1080 graphics card, achieving an [f1 score](https://deepai.org/machine-learning-glossary-and-terms/f-score) of 0.9951 on a held-out evaluation dataset. For more information about why neural networks train faster on graphics cards, [check here](https://towardsdatascience.com/why-deep-learning-uses-gpus-c61b399e93a0).)_

The tools in IDRT allow anyone with an existing dataset to train their own IDRT model. However, we recognize that many organizations who need to perform identity resolution do not have access to a large, high-quality dataset of prelabeled contact duplicates. As such, we have desigend IDRT so that it's easy to use a model that you did not train to match your own database of contacts. Our hope is that open-sourced and community-shared models will allow smaller organizations to take advantage of IDRT.

## How IDRT Works

All of the algorithms mentioned above compare two records directly. When you have a database of thousands or millions of records, you need to be able to efficiently apply a direct comarison to that dataset. Even a simple fuzzy matching algorithm might take a few milliseconds to compare two records. This is not likely to scale well in the naive algorithm of directly comparing each record to every other record. Neural networks are much more expensive to compute than other methods of directly comparing contacts. To efficiently scale to datasets with millions of records, we make use of a hybrid algorithm that uses neural networks and traditional search algorithms.

To facilitate this, IDRT actually uses two separate neural network models: a vector-encoder model and a classifier model. What these models do and how they are used in the identity resolution algorithm is explained below.

### Step 0: Gather the data

Before running IDRT, you should provide a query which will load the contacts from your SQL database. The exact fields (name, email, etc.) that you can provide depend on which fields the model you are using has been trained on. For more details about how to prepare your data for IDRT, see the documentation with the model release you are using.

(_TODO:_ Make sure to include a list of fields with any models that might get released...)

Crucially, your query must also provide an `updated_at` timestamp that represents the last time that contact was updated. This timestamp is not used directly in the duplicate detection algorithm, however it is crucial in the optimizations the algorithm performs to save work between runs.

### Step 1: Generate vector encodings for each contact

The first step of our algorithm is to generate a[vector representation](https://mathinsight.org/vector_introduction) of each contact in our database. This is what the first model, the __vector-encoder__ is trained to do. The vector-encoder model transforms a row of contact data like this:

```
John Doe, johndoe@email.com, 123-123-1212, AK
```

into a vector representation like this:

```
(0.21 1.12 -0.12 4.02 ...)
```

When every row of your data has been encoded, it is uploaded back into your database. A `contact_timestamp` column with the timestamp that the encoding was calculated is added. The table will look like this:

| pkey | contact_timestamp | x_0  | x_1  | x_2   | x_3  | ... |
|------|-------------------|------|------|-------|------|-----|
| 1234 | 2023-08-07 12:34  | 0.21 | 1.12 | -0.12 | 4.02 | ... |

The inclusion of the `contact_timestamp` column is a critical optimization for the algorithm! In subsequent runs of the algorithm, encodings will only be generated for existing contacts _with an `updated_at` timestamp after the `contact_timestmamp` for the existing encoding_. If you don't receive any new or updated contacts between runs of the alogrithm, it can skip Step 1 entirely.

### Step 2: Identify duplicate candidates from the vector encodings

In the field of computer science, a great deal of work has gone into optimizing searches among 