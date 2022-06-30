#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
import logging
import subprocess
from typing import Tuple

import click
from rxn.utilities.logging import setup_console_logger

from rxn_onmt_utils import __version__
from rxn_onmt_utils.rxn_models import defaults
from rxn_onmt_utils.rxn_models.onmt_train_command import OnmtTrainCommand
from rxn_onmt_utils.rxn_models.utils import ModelFiles, OnmtPreprocessedFiles

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command(context_settings=dict(show_default=True))
@click.option("--batch_size", default=defaults.BATCH_SIZE)
@click.option(
    "--data_weights",
    type=int,
    multiple=True,
    help="Weights of the different data sets for training. Only needed in a multi-task setting.",
)
@click.option("--dropout", default=defaults.DROPOUT)
@click.option("--heads", default=defaults.HEADS)
@click.option("--layers", default=defaults.LAYERS)
@click.option("--learning_rate", type=float, default=defaults.LEARNING_RATE)
@click.option(
    "--model_output_dir", type=str, required=True, help="Where to save the models"
)
@click.option("--no_gpu", is_flag=True, help="Run the training on CPU (slow!)")
@click.option(
    "--preprocess_dir",
    type=str,
    required=True,
    help="Directory with OpenNMT-preprocessed files",
)
@click.option("--rnn_size", default=defaults.RNN_SIZE)
@click.option("--seed", default=defaults.SEED)
@click.option("--train_num_steps", default=100000)
@click.option("--transformer_ff", default=defaults.TRANSFORMER_FF)
@click.option("--warmup_steps", default=defaults.WARMUP_STEPS)
@click.option("--word_vec_size", default=defaults.WORD_VEC_SIZE)
def main(
    batch_size: int,
    data_weights: Tuple[int, ...],
    dropout: float,
    heads: int,
    layers: int,
    learning_rate: float,
    model_output_dir: str,
    no_gpu: bool,
    preprocess_dir: str,
    rnn_size: int,
    seed: int,
    train_num_steps: int,
    transformer_ff: int,
    warmup_steps: int,
    word_vec_size: int,
) -> None:
    """Train an OpenNMT model.

    Multi-task training is also supported, if at least two
    `data_weights` parameters are given (Note: needs to be consistent with the
    rxn-onmt-preprocess command executed before training.
    """

    setup_console_logger()
    logger.info(f"Train RXN-OpenNMT model with rxn-onmt-utils, version {__version__}.")

    model_files = ModelFiles(model_output_dir)
    onmt_preprocessed_files = OnmtPreprocessedFiles(preprocess_dir)

    config_file = model_files.next_config_file()

    train_cmd = OnmtTrainCommand.train(
        batch_size=batch_size,
        data=onmt_preprocessed_files.preprocess_prefix,
        dropout=dropout,
        heads=heads,
        layers=layers,
        learning_rate=learning_rate,
        rnn_size=rnn_size,
        save_model=model_files.model_prefix,
        seed=seed,
        train_steps=train_num_steps,
        transformer_ff=transformer_ff,
        warmup_steps=warmup_steps,
        word_vec_size=word_vec_size,
        no_gpu=no_gpu,
        data_weights=data_weights,
    )

    # Write config file
    command_and_args = train_cmd.save_to_config_cmd(config_file)
    logger.info(f'Running command: {" ".join(command_and_args)}')
    _ = subprocess.run(command_and_args, check=True)

    # Actual training config file
    command_and_args = train_cmd.execute_from_config_cmd(config_file)
    logger.info(f'Running command: {" ".join(command_and_args)}')
    _ = subprocess.run(command_and_args, check=True)

    logger.info(
        f'Training successful. Models saved under "{str(model_files.model_dir)}".'
    )


if __name__ == "__main__":
    main()
