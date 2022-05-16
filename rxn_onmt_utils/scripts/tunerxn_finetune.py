#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
import logging
import subprocess
from typing import Tuple

import click
from rxn_utilities.logging_utilities import setup_console_logger

from rxn_onmt_utils.model_introspection import get_model_rnn_size, model_vocab_is_compatible
from rxn_onmt_utils.model_resize import ModelResizer
from rxn_onmt_utils.rxn_models import defaults
from rxn_onmt_utils.rxn_models.utils import (
    ModelFiles, OnmtPreprocessedFiles, extend_command_args_for_data_weights,
    extend_command_args_for_gpu
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command(context_settings=dict(show_default=True))
@click.option("--batch_size", default=defaults.BATCH_SIZE)
@click.option(
    '--data_weights',
    type=int,
    multiple=True,
    help='Weights of the different data sets for training. Only needed in a multi-task setting.'
)
@click.option("--dropout", default=defaults.DROPOUT)
@click.option("--learning_rate", type=float, default=0.06)
@click.option("--model_output_dir", type=str, required=True, help="Where to save the models")
@click.option("--no_gpu", is_flag=True, help='Run the training on CPU (slow!)')
@click.option(
    "--preprocess_dir",
    type=str,
    required=True,
    help="Directory with OpenNMT-preprocessed files",
)
@click.option("--seed", default=defaults.SEED)
@click.option("--train_from", type=str, required=True, help="Path to the model to start from")
@click.option("--train_num_steps", "--finetune_num_steps", default=100000)
@click.option("--warmup_steps", default=defaults.WARMUP_STEPS)
def main(
    batch_size: int,
    data_weights: Tuple[int, ...],
    dropout: float,
    learning_rate: float,
    model_output_dir: str,
    no_gpu: bool,
    preprocess_dir: str,
    seed: int,
    train_from: str,
    train_num_steps: int,
    warmup_steps: int,
) -> None:
    """Finetune an OpenNMT model."""

    setup_console_logger()

    model_files = ModelFiles(model_output_dir)
    onmt_preprocessed_files = OnmtPreprocessedFiles(preprocess_dir)

    if not model_vocab_is_compatible(train_from, onmt_preprocessed_files.vocab_file):
        # Extend the vocabulary of the base model based on the training data for
        # finetuning, and save its updated version in the new model directory.
        updated_base_model = model_files.model_dir / "updated_base_model.pt"
        resizer = ModelResizer(train_from)
        resizer.extend_vocab(onmt_preprocessed_files.vocab_file)
        resizer.save_checkpoint(updated_base_model)

        logger.info(
            f'The model checkpoint "{train_from}" needs additional vocabulary '
            f'tokens. The extended checkpoint was saved to "{updated_base_model}".'
        )
        train_from = str(updated_base_model)

    # In principle, the rnn_size should not be needed for finetuning. However,
    # when resetting the decay algorithm for the learning rate, this value
    # is necessary - and does not get it from the model checkpoint (OpenNMT bug).
    rnn_size = get_model_rnn_size(train_from)
    logger.info(f'Loaded the value of rnn_size from the model: {rnn_size}.')

    config_file = model_files.next_config_file()

    # yapf: disable
    command_and_args = [
        str(e) for e in [
            'onmt_train',
            '-save_config', config_file,
            '-accum_count', '4',
            '-adam_beta1', '0.9',
            '-adam_beta2', '0.998',
            '-batch_size', batch_size,
            '-batch_type', 'tokens',
            '-data', onmt_preprocessed_files.preprocess_prefix,
            '-decay_method', 'noam',
            '-dropout', dropout,
            '-keep_checkpoint', '20',
            '-label_smoothing', '0.0',
            '-learning_rate', learning_rate,
            '-max_generator_batches', '32',
            '-max_grad_norm', '0',
            '-normalization', 'tokens',
            '-optim', 'adam',
            '-report_every', '1000',
            '-reset_optim', 'all',
            '-rnn_size', rnn_size,
            '-save_checkpoint_steps', '5000',
            '-save_model', model_files.model_prefix,
            '-seed', seed,
            '-train_from', train_from,
            '-train_steps', train_num_steps,
            '-valid_batch_size', '8',
            '-warmup_steps', warmup_steps,
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
    command_and_args = ['onmt_train', '-config', str(config_file)]
    logger.info(f'Running command: {" ".join(command_and_args)}')
    _ = subprocess.run(command_and_args, check=True)

    logger.info(f'Fine-tuning successful. Models saved under "{str(model_files.model_dir)}".')


if __name__ == "__main__":
    main()
