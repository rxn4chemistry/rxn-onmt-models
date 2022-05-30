# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2021
# ALL RIGHTS RESERVED

import logging
from typing import Tuple

import click

from rxn_onmt_utils.torch_utils import set_num_threads
from rxn_onmt_utils.translator import Translator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command()
@click.option(
    "--models",
    "-m",
    multiple=True,
    help="Translation model file. If multiple are given, will be an ensemble model.",
)
@click.option("--src_file", "-s", required=True, help="File to translate")
@click.option("--output_file", "-o", required=True, help="Where to save translation")
@click.option("--num_threads", "-n", default=4, help="Number of CPU threads to use")
def translate(
    models: Tuple[str, ...], src_file: str, output_file: str, num_threads: int
):
    """Translate lines of a file with one or several OpenNMT model(s)."""
    set_num_threads(num_threads)

    translator = Translator.from_model_path(model_path=models)

    with open(src_file, "rt") as f:
        sentences = [line.strip() for line in f]

    translations = translator.translate_sentences(sentences)

    with open(output_file, "wt") as f:
        for t in translations:
            f.write(f"{t}\n")


if __name__ == "__main__":
    translate()
