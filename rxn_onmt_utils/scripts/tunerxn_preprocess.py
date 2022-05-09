#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
import subprocess
from typing import Optional

import click

from rxn_onmt_utils.from_tunerxn.utils import RxnPreprocessingFiles, OnmtPreprocessedFiles


@click.command()
@click.option(
    '--input_dir', type=str, required=True, help='Directory containing the output of prepare-data'
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
def preprocess(input_dir: str, output_dir: str, model_task: str, vocab: Optional[str]) -> None:
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
    """

    data_files = RxnPreprocessingFiles(input_dir)
    onmt_preprocessed_files = OnmtPreprocessedFiles(output_dir)

    train_src = data_files.get_tokenized_src_file('train', model_task)
    train_tgt = data_files.get_tokenized_tgt_file('train', model_task)
    valid_src = data_files.get_tokenized_src_file('valid', model_task)
    valid_tgt = data_files.get_tokenized_tgt_file('valid', model_task)

    # yapf: disable
    command_and_args = [
        'onmt_preprocess',
        '-train_src', str(train_src),
        '-train_tgt', str(train_tgt),
        '-valid_src', str(valid_src),
        '-valid_tgt', str(valid_tgt),
        '-save_data', str(onmt_preprocessed_files.preprocess_prefix),
        '-src_seq_length', '3000',
        '-tgt_seq_length', '3000',
        '-src_vocab_size', '3000',
        '-tgt_vocab_size', '3000',
        '-share_vocab',
        '-overwrite',
    ]
    # yapf: enable
    if vocab is not None:
        command_and_args.extend(['-src_vocab', vocab, '-tgt_vocab', vocab])

    print('Running command:', ' '.join(command_and_args))
    _ = subprocess.run(command_and_args, check=True)


if __name__ == "__main__":
    preprocess()
