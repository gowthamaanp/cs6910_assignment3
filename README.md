<div style="text-align: justify;">

# CS6910 Fundamentals of Deep Learning - Assignment 3

This repository contains all files for the third assignment of the CS6910 - Fundamentals of Deep Learning course at IIT Madras.

## Contents

- [Task](#task)
- [Submission](#submission)
- [Dataset](#dataset)
- [Implementation Details](#implementation-details)
- [Tools and Libraries Used](#tools-and-libraries-used)
  - [Installation](#installation)
- [Usage](#usage)

## Task

The task is to use recurrent neural networks to build a transliteration system

## Submission

My WandB project: https://wandb.ai/ed23s037/CS6910_AS3

My WandB report: https://api.wandb.ai/links/ed23s037/8y7zo8uw

## Dataset

The dataset is a sample of the [Aksharantar dataset](https://drive.google.com/file/d/1tGIO4-IPNtxJ6RQMmykvAfY_B0AaLY5A/view?usp=drive_link) released by AI4Bharat. This dataset contains pairs of words in the native script and its corresponding transliteration in the Latin script.

## Implementation Details

In this section, we delve into the implementation of the transliteration system, detailing the structure and functionalities of each component.

## Tools and Libraries Used

The implementation of the project utilizes the following tools and libraries:

- **Python 3.10.1**: The core programming language used for implementing all aspects of the project.

- **WandB (Weights and Biases)**: WandB is utilized for running experiments, hyperparameter tuning, visualization, and more.

- **Pytorch 2.0.0**: Pytorch framework is used to implement the RNN Encoder and Decoder network.

- **Pandas 1.5.3**: Pandas library is used to read the data from the .csv dataset files.

### Installation

All the above packages are listed in the `requirements.txt` file. To install them, simply execute the following command:

```sh
$ pip install -r requirements.txt
```

By installing the listed packages, you can ensure that all necessary dependencies are met for running the project smoothly.

## Usage

The main.py file is the entry point to all the functions. Run the following command to see the options:

```sh
$ python main.py
```

With this file you can train, evaluate, infer(single word) and sweep the model. Go through the file for more details on parameters for every configuration. The common template is the following:

```sh
$ python main.py <func> <parameters>
```

For example to get help on training the model use the following command:

```sh
$ python main.py train -h
```

</div>
