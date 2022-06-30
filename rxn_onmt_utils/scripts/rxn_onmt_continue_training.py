#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
import logging
import subprocess
from typing import Optional, Tuple

import click
from rxn.utilities.logging import setup_console_logger

from rxn_onmt_utils import __version__
from rxn_onmt_utils.model_introspection import (
    get_model_dropout,
    get_model_seed,
    model_vocab_is_compatible,
)
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
@click.option(
    "--train_from",
    type=str,
    help=(
        "Model to continue training from. If not specified, "
        "the last checkpoint from model_output_dir will be taken."
    ),
)
@click.option(
    "--train_num_steps",
    default=100000,
    help="Number of steps, including steps from the initial training run.",
)
def main(
    batch_size: int,
    data_weights: Tuple[int, ...],
    model_output_dir: str,
    no_gpu: bool,
    preprocess_dir: str,
    train_from: Optional[str],
    train_num_steps: int,
) -> None:
    """Continue training for an OpenNMT model.

    Multi-task training is also supported, if at least two
    `data_weights` parameters are given (Note: needs to be consistent with the
    rxn-onmt-preprocess command executed before training.
    """

    setup_console_logger()
    logger.info(
        "Continue training of RXN-OpenNMT model with rxn-onmt-utils, "
        f"version {__version__}."
    )

    model_files = ModelFiles(model_output_dir)
    onmt_preprocessed_files = OnmtPreprocessedFiles(preprocess_dir)

    if train_from is None:
        train_from = str(model_files.get_last_checkpoint())
    logger.info(f"Training will be continued from {train_from}")

    if not model_vocab_is_compatible(train_from, onmt_preprocessed_files.vocab_file):
        raise RuntimeError(
            "The vocabularies are not compatible. It is not advised to continue training."
        )

    config_file = model_files.next_config_file()
    dropout = get_model_dropout(train_from)
    seed = get_model_seed(train_from)

    train_cmd = OnmtTrainCommand.continue_training(
        batch_size=batch_size,
        data=onmt_preprocessed_files.preprocess_prefix,
        dropout=dropout,
        save_model=model_files.model_prefix,
        seed=seed,
        train_from=train_from,
        train_steps=train_num_steps,
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
