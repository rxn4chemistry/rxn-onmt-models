#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
import logging
import subprocess
from typing import List, Tuple

import click
from rxn_utilities.logging_utilities import setup_console_logger

import rxn_onmt_utils.rxn_models.defaults as defaults
from rxn_onmt_utils.rxn_models.utils import (
    extend_command_args_for_gpu, extend_command_args_for_data_weights, ModelFiles,
    OnmtPreprocessedFiles
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command(context_settings=dict(show_default=True))
@click.option("--batch_size", default=defaults.BATCH_SIZE)
@click.option(
    "--data_weights",
    type=int,
    multiple=True,
    help='Weights of the different data sets for training. Only needed in a multi-task setting.'
)
@click.option("--dropout", default=defaults.DROPOUT)
@click.option("--model_output_dir", type=str, required=True, help="Where to save the models")
@click.option("--no_gpu", is_flag=True, help='Run the training on CPU (slow!)')
@click.option(
    "--preprocess_dir",
    type=str,
    required=True,
    help="Directory with OpenNMT-preprocessed files",
)
@click.option("--seed", default=defaults.SEED)
@click.option(
    "--train_num_steps",
    default=100000,
    help="Number of steps, including steps from the initial training run."
)
def main(
    batch_size: int,
    data_weights: Tuple[int, ...],
    dropout: float,
    model_output_dir: str,
    no_gpu: bool,
    preprocess_dir: str,
    seed: int,
    train_num_steps: int,
) -> None:
    """Continue training for an OpenNMT model.

    Multi-task training is also supported, if at least two
    `data_weights` parameters are given (Note: needs to be consistent with the
    rxn-onmt-preprocess command executed before training.
    """

    setup_console_logger()

    model_files = ModelFiles(model_output_dir)
    onmt_preprocessed_files = OnmtPreprocessedFiles(preprocess_dir)

    train_from = model_files.get_last_checkpoint()
    logger.info(f'Training will be continued from {train_from}')

    # yapf: disable
    command_and_args: List[str] = [
        str(e) for e in [
            'onmt_train',
            '-save_config', model_files.config_file,
            '-accum_count', '4',
            '-batch_size', batch_size,
            '-batch_type', 'tokens',
            '-data', onmt_preprocessed_files.preprocess_prefix,
            '-dropout', dropout,
            '-keep_checkpoint', '20',
            '-label_smoothing', '0.0',
            '-max_generator_batches', '32',
            '-normalization', 'tokens',
            '-report_every', '1000',
            '-reset_optim', 'none',
            '-save_checkpoint_steps', '5000',
            '-save_model', model_files.model_prefix,
            '-seed', seed,
            '-train_from', train_from,
            '-train_steps', train_num_steps,
            '-valid_batch_size', '8',
        ]
    ]
    # yapf: enable

    extend_command_args_for_gpu(command_and_args, no_gpu=no_gpu)
    extend_command_args_for_data_weights(command_and_args, data_weights=data_weights)

    # Write config file
    command_and_args = [str(v) for v in command_and_args]
    logger.info(f'Running command: {" ".join(command_and_args)}')
    _ = subprocess.run(command_and_args, check=True)

    # Actual training config file
    command_and_args = ['onmt_train', '-config', str(model_files.config_file)]
    logger.info(f'Running command: {" ".join(command_and_args)}')
    _ = subprocess.run(command_and_args, check=True)

    logger.info(f'Training successful. Models saved under {str(model_files.model_dir)}')


if __name__ == "__main__":
    main()
