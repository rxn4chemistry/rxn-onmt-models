# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2021
# ALL RIGHTS RESERVED

import logging
from typing import Tuple

import click

from rxn_onmt_utils.model_resize import ModelResizer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command()
@click.option(
    '--model_path', '-m', required=True, help='Pretrained ONMT model.'
)
@click.option(
    '--vocab_path', '-v', required=True, help='Vocab for finetuning.'
)
@click.option(
    '--output_path',
    '-o',
    required=True,
    help='Where to save the resized model.'
)
def extend_model_vocab(model_path: str, vocab_path: str, output_path: str):
    """Extend model vocab, resize and initialise additional weights."""

    resizer = ModelResizer(model_path)

    resizer.extend_vocab(vocab_path)

    resizer.save_checkpoint(output_path)


if __name__ == "__main__":
    extend_model_vocab()
