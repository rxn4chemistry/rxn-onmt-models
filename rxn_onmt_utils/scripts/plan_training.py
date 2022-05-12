#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED

import click

import rxn_onmt_utils.rxn_models.defaults as defaults


def prepare_data_cmd(data_txt: str, data_dir: str, prepare_seed: int) -> str:
    command = f'rxn-prepare-data --input_data {data_txt} --output_dir {data_dir} '
    if prepare_seed != defaults.SEED:
        command += f'--split_seed {prepare_seed} '
    return command


@click.command()
def main() -> None:
    """Interactive program to plan the training of RXN OpenNMT models.

    It will ask a user for the values needed for training, and then print all
    the commands to be executed.
    """

    print('Interactive program to plan the training of RXN OpenNMT models.')
    print('NOTE: Please avoid using paths with whitespaces.')

    model_task = click.prompt(
        'Please enter a valid integer', type=click.Choice(['forward', 'retro'])
    )

    on_gpu = click.confirm('GPU available?', default=True)
    is_multi_task = click.confirm(
        'Do you want to train on multiple data sets (multi-task)?', default=False
    )

    main_data_txt = click.prompt('Path to the main data set (TXT)', type=str)
    main_data_dir = click.prompt('Where to save the main processed data set', type=str)

    # Get all the paths to data
    data_txts = [main_data_txt]
    data_dirs = [main_data_dir]
    data_weights = []
    if is_multi_task:
        number_additional_datasets = click.prompt(
            'Number of additional datasets', type=click.IntRange(min=1)
        )
        for i in range(number_additional_datasets):
            data_txt = click.prompt(f'Path to the additional data set (TXT) no {i+1}', type=str)
            data_dir = click.prompt(f'Where to save the processed data set no {i+1}', type=str)
            data_txts.append(data_txt)
            data_dirs.append(data_dir)
        for data_txt in data_txts:
            weight = click.prompt(
                f'Training weight for data set in "{data_txt}"', type=click.IntRange(min=1)
            )
            data_weights.append(weight)

    preprocess_seed = click.prompt('Seed for data preprocessing', type=int, default=defaults.SEED)

    onmt_preprocessed = click.prompt('Where to save the OpenNMT-preprocessed data', type=str)
    onmt_models = click.prompt('Where to save the OpenNMT models', type=str)

    batch_size = click.prompt('Batch size', type=int, default=defaults.BATCH_SIZE)
    train_steps = click.prompt('Number of training steps', type=int, default=100000)
    learning_rate = click.prompt('Learning rate', type=float, default=defaults.LEARNING_RATE)

    dropout = click.prompt('Dropout', type=float, default=defaults.DROPOUT)
    heads = click.prompt('Number of transformer heads', type=int, default=defaults.HEADS)
    layers = click.prompt('Number of layers', type=int, default=defaults.LAYERS)
    rnn_size = click.prompt('RNN size', type=int, default=defaults.RNN_SIZE)
    transformer_ff = click.prompt(
        'Size of hidden transformer feed-forward', type=int, default=defaults.TRANSFORMER_FF
    )
    word_vec_size = click.prompt('Word embedding size', type=int, default=defaults.WORD_VEC_SIZE)
    warmup_steps = click.prompt('Number of warmup steps', type=int, default=defaults.WARMUP_STEPS)
    train_seed = click.prompt('See for training', type=int, default=defaults.SEED)

    preprocess = (
        'rxn-onmt-preprocess '
        f'--input_dir {main_data_dir} '
        f'--output_dir {onmt_preprocessed} '
        f'--model_task {model_task} '
    )
    if is_multi_task:
        for data_dir in data_dirs[1:]:
            preprocess += f'--additional_data {data_dir} '

    train = (
        'rxn-onmt-train '
        f'--model_output_dir {onmt_models} '
        f'--preprocess_dir {onmt_preprocessed} '
        f'--train_num_steps {train_steps} '
    )
    if batch_size != defaults.BATCH_SIZE:
        train += f'--batch_size {batch_size} '
    if dropout != defaults.DROPOUT:
        train += f'--dropout {dropout} '
    if heads != defaults.HEADS:
        train += f'--heads {heads} '
    if layers != defaults.LAYERS:
        train += f'--layers {layers} '
    if learning_rate != defaults.LEARNING_RATE:
        train += f'--learning_rate {learning_rate} '
    if rnn_size != defaults.RNN_SIZE:
        train += f'--rnn_size {rnn_size} '
    if train_seed != defaults.SEED:
        train += f'--seed {train_seed} '
    if transformer_ff != defaults.TRANSFORMER_FF:
        train += f'--transformer_ff {transformer_ff} '
    if warmup_steps != defaults.WARMUP_STEPS:
        train += f'--warmup_steps {warmup_steps} '
    if word_vec_size != defaults.WORD_VEC_SIZE:
        train += f'--word_vec_size {word_vec_size} '
    if is_multi_task:
        for weight in data_weights:
            train += f'--data_weights {weight} '

    continue_training = (
        'rxn-onmt-continue-training '
        f'--model_output_dir {onmt_models} '
        f'--preprocess_dir {onmt_preprocessed} '
        f'--train_num_steps {train_steps} '
    )
    if batch_size != defaults.BATCH_SIZE:
        continue_training += f'--batch_size {batch_size} '
    if dropout != defaults.DROPOUT:
        continue_training += f'--dropout {dropout} '
    if train_seed != defaults.SEED:
        continue_training += f'--seed {train_seed} '
    if is_multi_task:
        for weight in data_weights:
            continue_training += f'--data_weights {weight} '

    if not on_gpu:
        train += '--no_gpu '
        continue_training += '--no_gpu '

    print('Here are the commands to launch a training with RXN:\n')
    print('# 1) Prepare the data (standardization, filtering, etc.)')
    for data_txt, data_dir in zip(data_txts, data_dirs):
        print(prepare_data_cmd(data_txt, data_dir, preprocess_seed))
    print()
    print(f'# 2) Preprocess the data with OpenNMT\n{preprocess}\n')
    print(f'# 3) Train the model\n{train}\n')
    print(f'# 4) If necessary: continue training\n{continue_training}')


if __name__ == "__main__":
    main()
