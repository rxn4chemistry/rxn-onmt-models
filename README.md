# rxn_onmt_training

Utilities related to the training of RXN models with OpenNMT.

## RXN model training

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
