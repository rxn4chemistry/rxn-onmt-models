# rxn_onmt_utils

[![Build Status](https://travis.ibm.com/rxn/rxn_onmt_utils.svg?token=zJxfB9t9kgVLYHLdp5pG&branch=develop)](https://travis.ibm.com/rxn/rxn_onmt_utils)

Utilities related to the use of OpenNMT.

The repository contains both (i) ONMT-related code that is independent of the RXN models, as well as (ii) code specific to chemistry tasks.
The files for (ii) are mainly located under the `rxn_models` directory. Some of this is independent of OpenNMT and may be moved elsewhere at a later stage.


## RXN model training

Example of usage for training RXN models

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
