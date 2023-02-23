import logging
from typing import Tuple

import click
from rxn.onmt_utils import __version__ as onmt_utils_version
from rxn.onmt_utils.model_introspection import (
    get_model_rnn_size,
    model_vocab_is_compatible,
)
from rxn.onmt_utils.model_resize import ModelResizer
from rxn.onmt_utils.train_command import OnmtTrainCommand
from rxn.utilities.logging import setup_console_and_file_logger

from rxn.onmt_models import __version__ as onmt_models_version
from rxn.onmt_models import defaults
from rxn.onmt_models.training_files import ModelFiles, OnmtPreprocessedFiles
from rxn.onmt_models.utils import log_file_name_from_time, run_command

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
@click.option("--learning_rate", type=float, default=0.06)
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
@click.option("--seed", default=defaults.SEED)
@click.option(
    "--train_from", type=str, required=True, help="Path to the model to start from"
)
@click.option("--train_num_steps", "--finetune_num_steps", default=100000)
@click.option("--warmup_steps", default=defaults.WARMUP_STEPS)
@click.option("--report_every", default=1000)
@click.option("--save_checkpoint_steps", default=5000)
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
    report_every: int,
    save_checkpoint_steps: int,
) -> None:
    """Finetune an OpenNMT model."""

    # Set up paths
    model_files = ModelFiles(model_output_dir)
    onmt_preprocessed_files = OnmtPreprocessedFiles(preprocess_dir)

    # Set up the logs
    log_file = model_files.model_dir / log_file_name_from_time("rxn-onmt-finetune")
    setup_console_and_file_logger(log_file)

    logger.info("Fine-tuning RXN model.")
    logger.info(f"rxn-onmt-utils version: {onmt_utils_version}. ")
    logger.info(f"rxn-onmt-models version: {onmt_models_version}. ")

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
    logger.info(f"Loaded the value of rnn_size from the model: {rnn_size}.")

    config_file = model_files.next_config_file()

    train_cmd = OnmtTrainCommand.finetune(
        batch_size=batch_size,
        data=onmt_preprocessed_files.preprocess_prefix,
        dropout=dropout,
        learning_rate=learning_rate,
        rnn_size=rnn_size,
        save_model=model_files.model_prefix,
        seed=seed,
        train_from=train_from,
        train_steps=train_num_steps,
        warmup_steps=warmup_steps,
        no_gpu=no_gpu,
        data_weights=data_weights,
        report_every=report_every,
        save_checkpoint_steps=save_checkpoint_steps,
    )

    # Write config file
    command_and_args = train_cmd.save_to_config_cmd(config_file)
    run_command(command_and_args)

    # Actual training config file
    command_and_args = train_cmd.execute_from_config_cmd(config_file)
    run_command(command_and_args)

    logger.info(
        f'Fine-tuning successful. Models saved under "{str(model_files.model_dir)}".'
    )


if __name__ == "__main__":
    main()
