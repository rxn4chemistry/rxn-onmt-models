import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import click
from rxn.chemutils.tokenization import ensure_tokenized_file
from rxn.onmt_utils import __version__ as onmt_utils_version
from rxn.onmt_utils.train_command import preprocessed_id_names
from rxn.utilities.files import (
    PathLike,
    count_lines,
    dump_list_to_file,
    load_list_from_file,
)
from rxn.utilities.logging import setup_console_and_file_logger

from rxn.onmt_models import __version__ as onmt_models_version
from rxn.onmt_models import defaults
from rxn.onmt_models.training_files import OnmtPreprocessedFiles, RxnPreprocessingFiles
from rxn.onmt_models.utils import log_file_name_from_time, run_command

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def determine_train_dataset(
    data: RxnPreprocessingFiles, model_task: str
) -> Tuple[Path, Path]:
    """
    Get the paths to the src and tgt dataset, trying to get the augmented
    one if it exists.

    Args:
        data: info about training files.
        model_task: model task.

    Returns:
        Tuple for the src and tgt files (the augmented ones if possible).
    """
    src = data.get_src_file("train", model_task)
    tgt = data.get_tgt_file("train", model_task)

    augmented_src = data.augmented(src)
    augmented_tgt = data.augmented(tgt)
    if augmented_src.exists() and augmented_tgt.exists():
        logger.info(f'Found augmented train split in "{data.processed_data_dir}"')
        src = augmented_src
        tgt = augmented_tgt

    return src, tgt


@click.command()
@click.option(
    "--input_dir",
    type=str,
    required=True,
    help="Directory containing the output of prepare-data for the main data set.",
)
@click.option(
    "--additional_data",
    type=str,
    multiple=True,
    help="Directory containing the output of prepare-data for the additional data sets.",
)
@click.option(
    "--output_dir",
    type=str,
    required=True,
    help="Where to save the preprocessed OpenNMT files.",
)
@click.option(
    "--model_task", type=click.Choice(["forward", "retro", "context"]), required=True
)
@click.option(
    "--truncated_valid_size",
    default=defaults.VALIDATION_TRUNCATE_SIZE,
    help=(
        "Number of samples from the validation set to consider for reporting the accuracy "
        "on the validation set. From experiences, taking values larger than 10k just "
        "leads to longer training times without much gain. Use -1 for no truncation."
    ),
)
@click.option(
    "--truncation_shuffling_seed",
    default=defaults.SEED,
    help="Random seed to use for shuffling the validation reactions before truncation.",
)
@click.option(
    "--vocab",
    type=str,
    help=(
        "Token vocabulary file (one token per line). Required only in order "
        "to add tokens not in the dataset when training the base model."
    ),
)
def main(
    input_dir: str,
    additional_data: Tuple[str, ...],
    output_dir: str,
    model_task: str,
    truncated_valid_size: int,
    truncation_shuffling_seed: int,
    vocab: Optional[str],
) -> None:
    """Preprocess the training files for OpenNMT models (wraps onmt_preprocess).

    The input_dir must contain the following files:
        data.processed.train.precursors_tokens
        data.processed.train.products_tokens
        data.processed.validation.precursors_tokens
        data.processed.validation.products_tokens

    The script will generate the following files in output_dir:
        preprocessed.train.0.pt
        preprocessed.valid.0.pt
        preprocessed.vocab.pt
        ... (and additional indices for train and valid if the dataset is large)

    Preprocessing data for multitask training is also supported, if at least one
    `additional_data` parameter is given.
    """

    # Set up the paths
    main_data_files = RxnPreprocessingFiles(input_dir)
    onmt_preprocessed_files = OnmtPreprocessedFiles(output_dir)

    # Set up the logs
    log_file = onmt_preprocessed_files.preprocessed_dir / log_file_name_from_time(
        "rxn-onmt-preprocess"
    )
    setup_console_and_file_logger(log_file)

    logger.info("Preprocess data for RXN-OpenNMT models.")
    logger.info(f"rxn-onmt-utils version: {onmt_utils_version}. ")
    logger.info(f"rxn-onmt-models version: {onmt_models_version}. ")

    train_src, train_tgt = determine_train_dataset(main_data_files, model_task)
    valid_src: PathLike = main_data_files.get_src_file("valid", model_task)
    valid_tgt: PathLike = main_data_files.get_tgt_file("valid", model_task)

    train_srcs: List[PathLike] = [train_src]
    train_tgts: List[PathLike] = [train_tgt]

    for i, additional_data_path in enumerate(additional_data, 1):
        data_files = RxnPreprocessingFiles(additional_data_path)
        src, tgt = determine_train_dataset(data_files, model_task)
        train_srcs.append(src)
        train_tgts.append(tgt)

    if truncated_valid_size != -1 and count_lines(valid_src) > truncated_valid_size:
        logger.info(
            f"The validation set will be truncated to {truncated_valid_size} lines."
        )

        # Load all samples and put in list of src-tgt tuples
        valid_src_tgt = list(
            zip(load_list_from_file(valid_src), load_list_from_file(valid_tgt))
        )

        # Shuffle the samples and truncate
        random.seed(truncation_shuffling_seed)
        random.shuffle(valid_src_tgt)
        valid_src_tgt = valid_src_tgt[:truncated_valid_size]

        # Write to new files
        valid_src = onmt_preprocessed_files.preprocessed_dir / "truncated_valid_src.txt"
        valid_tgt = onmt_preprocessed_files.preprocessed_dir / "truncated_valid_tgt.txt"
        dump_list_to_file((src for src, _ in valid_src_tgt), valid_src)
        dump_list_to_file((tgt for _, tgt in valid_src_tgt), valid_tgt)

        logger.info(
            f'The truncated validation set was saved to "{valid_src}" and "{valid_tgt}".'
        )

    # Tokenize all the files if necessary
    train_srcs = [ensure_tokenized_file(f) for f in train_srcs]
    train_tgts = [ensure_tokenized_file(f) for f in train_tgts]
    valid_src = ensure_tokenized_file(valid_src)
    valid_tgt = ensure_tokenized_file(valid_tgt)

    # yapf: disable
    command_and_args = [
        str(e) for e in [
            'onmt_preprocess',
            '-train_src', *train_srcs,
            '-train_tgt', *train_tgts,
            '-valid_src', valid_src,
            '-valid_tgt', valid_tgt,
            '-save_data', onmt_preprocessed_files.preprocess_prefix,
            '-src_seq_length', 3000,
            '-tgt_seq_length', 3000,
            '-src_vocab_size', 3000,
            '-tgt_vocab_size', 3000,
            '-share_vocab',
            '-overwrite',
        ]
    ]
    # yapf: enable
    if vocab is not None:
        command_and_args.extend(["-src_vocab", vocab, "-tgt_vocab", vocab])

    if additional_data:
        train_ids = preprocessed_id_names(len(additional_data))
        command_and_args.extend(["-train_ids", *train_ids])

    run_command(command_and_args)


if __name__ == "__main__":
    main()
