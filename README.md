# RXN package for OpenNMT-based models

[![Actions tests](https://github.com/rxn4chemistry/rxn-onmt-models/actions/workflows/tests.yaml/badge.svg)](https://github.com/rxn4chemistry/rxn-onmt-models/actions)

This repository contains a Python package and associated scripts for training reaction models based on the OpenNMT library.
The repository is built on top of other RXN packages; see our other repositories [`rxn-utilities`](https://github.com/rxn4chemistry/rxn-utilities), [`rxn-chemutils`](https://github.com/rxn4chemistry/rxn-chemutils), and [`rxn-onmt-utils`](https://github.com/rxn4chemistry/rxn-onmt-utils).

For the evaluation of trained models, see the [`rxn-metrics`](https://github.com/rxn4chemistry/rxn-metrics) repository.

The documentation can be found [here](https://rxn4chemistry.github.io/rxn-onmt-models/).

This repository was produced through a collaborative project involving IBM Research Europe and Syngenta.

## System Requirements

This package is supported on all operating systems.
It has been tested on the following systems:
+ macOS: Big Sur (11.1)
+ Linux: Ubuntu 18.04.4

A Python version of 3.6, 3.7, or 3.8 is recommended.
Python versions 3.9 and above are not expected to work due to compatibility with the selected version of OpenNMT.

## Installation guide

The package can be installed from Pypi:
```bash
pip install rxn-onmt-models[rdkit]
```
You can leave out `[rdkit]` if RDKit is already available in your environment.

For local development, the package can be installed with:
```bash
pip install -e ".[dev,rdkit]"
```

## Training models.

Example of usage for training RXN models

### The easy way

Simply execute the interactive program `rxn-plan-training` in your terminal and follow the instructions.

### The complicated way

0. Optional: set shell variables, to be used in the commands later on.

```shell
MODEL_TASK="forward"

# Existing TXT files
DATA_1="/path/to/data_1.txt"
DATA_2="/path/to/data_2.txt"
DATA_3="/path/to/data_3.txt"

# Where to put the processed data
DATA_DIR_1="/path/to/processed_data_1"
DATA_DIR_2="/path/to/processed_data_2"
DATA_DIR_3="/path/to/processed_data_3"

# Where to save the ONMT-preprocessed data
PREPROCESSED="/path/to/onmt-preprocessed"

# Where to save the models
MODELS="/path/to/models"
MODELS_FINETUNED="/path/to/models_finetuned"
```

1. Prepare the data (standardization, filtering, etc.)

```shell
rxn-prepare-data --input_data $DATA_1 --output_dir $DATA_DIR_1
```

2. Preprocess the data with OpenNMT

```shell
rxn-onmt-preprocess --input_dir $DATA_DIR_1 --output_dir $PREPROCESSED --model_task $MODEL_TASK
```

3. Train the model (here with small parameter values, to make it fast on CPU for testing).

```shell
rxn-onmt-train --model_output_dir $MODELS --preprocess_dir $PREPROCESSED_SINGLE --train_num_steps 10 --batch_size 4 --heads 2 --layers 2 --transformer_ff 512 --no_gpu
```

### Multi-task training

For multi-task training, the process is similar. 
We need to prepare also the second data set; in addition, the OpenNMT preprocessing and training take additional arguments.
To sum up:

```shell
rxn-prepare-data --input_data $DATA_1 --output_dir $DATA_DIR_1
rxn-prepare-data --input_data $DATA_2 --output_dir $DATA_DIR_2
rxn-prepare-data --input_data $DATA_2 --output_dir $DATA_DIR_3
rxn-onmt-preprocess --input_dir $DATA_DIR_1 --output_dir $PREPROCESSED --model_task $MODEL_TASK \
  --additional_data $DATA_DIR_2 --additional_data $DATA_DIR_3
rxn-onmt-train --model_output_dir $MODELS --preprocess_dir $PREPROCESSED --train_num_steps 30 --batch_size 4 --heads 2 --layers 2 --transformer_ff 256 --no_gpu \
  --data_weights 1 --data_weights 3 --data_weights 4
```

### Continuing the training

Continuing training is possible (for both single-task and multi-task); it needs fewer parameters:
```shell
rxn-onmt-continue-training --model_output_dir $MODELS --preprocess_dir $PREPROCESSED --train_num_steps 30 --batch_size 4 --no_gpu \
  --data_weights 1 --data_weights 3 --data_weights 4
```

### Fine-tuning

Fine-tuning is in principle similar to continuing the training. 
The main differences are the potential occurrence of new tokens, as well as the optimizer being reset.
There is a dedicated command for fine-tuning. For example:
```shell
rxn-onmt-finetune --model_output_dir $MODELS_FINETUNED --preprocess_dir $PREPROCESSED --train_num_steps 20 --batch_size 4 --no_gpu \
  --train_from $MODELS/model_step_30.pt
```
The syntax is very similar to `rxn-onmt-train` and `rxn-onmt-continue-training`.
This is compatible both with single-task and multi-task.
