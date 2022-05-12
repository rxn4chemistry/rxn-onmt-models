#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
import logging
import subprocess
from typing import Optional, Tuple

import click
from rxn_utilities.logging_utilities import setup_console_logger

from rxn_onmt_utils.from_tunerxn.utils import RxnPreprocessingFiles, OnmtPreprocessedFiles
from rxn_onmt_utils.rxn_models.utils import preprocessed_id_names

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command()
@click.option(
    '--input_dir',
    type=str,
    required=True,
    help='Directory containing the output of prepare-data for the main data set.'
)
@click.option(
    '--additional_data',
    type=str,
    multiple=True,
    help='Directory containing the output of prepare-data for the additional data sets.'
)
@click.option(
    '--output_dir', type=str, required=True, help='Where to save the preprocessed OpenNMT files.'
)
@click.option('--model_task', type=click.Choice(['forward', 'retro']), required=True)
@click.option(
    '--vocab',
    type=str,
    help=(
        'Token vocabulary file (one token per line). Required only in order '
        'to add tokens not in the dataset when training the base model.'
    )
)
def main(
    input_dir: str, additional_data: Tuple[str, ...], output_dir: str, model_task: str,
    vocab: Optional[str]
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

    Preprocessing data for multi-task training is also supported, if at least one
    `additional_data` parameter is given.
    """
    setup_console_logger()

    main_data_files = RxnPreprocessingFiles(input_dir)
    onmt_preprocessed_files = OnmtPreprocessedFiles(output_dir)

    train_src = main_data_files.get_tokenized_src_file('train', model_task)
    train_tgt = main_data_files.get_tokenized_tgt_file('train', model_task)
    valid_src = main_data_files.get_tokenized_src_file('valid', model_task)
    valid_tgt = main_data_files.get_tokenized_tgt_file('valid', model_task)

    train_srcs = [train_src]
    train_tgts = [train_tgt]

    for i, additional_data_path in enumerate(additional_data, 1):
        data_files = RxnPreprocessingFiles(additional_data_path)
        train_srcs.append(data_files.get_tokenized_src_file('train', model_task))
        train_tgts.append(data_files.get_tokenized_tgt_file('train', model_task))

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
        command_and_args.extend(['-src_vocab', vocab, '-tgt_vocab', vocab])

    if additional_data:
        train_ids = preprocessed_id_names(len(additional_data))
        command_and_args.extend(['-train_ids', *train_ids])

    logger.info(f'Running command: {" ".join(command_and_args)}')
    try:
        output = subprocess.check_output(command_and_args)
    except subprocess.CalledProcessError as e:
        logger.exception('Error during OpenNMT preprocessing')
        raise SystemExit('Error during OpenNMT preprocessing') from e

    logger.info(f'Command ran successfully. Output:\n{output.decode("UTF-8")}')


if __name__ == "__main__":
    main()
