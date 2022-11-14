#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
import json
import logging
from shutil import copy

import click
from rxn.chemutils.reaction_equation import (
    ReactionEquation,
    canonicalize_compounds,
    sort_compounds,
)
from rxn.chemutils.reaction_smiles import parse_any_reaction_smiles
from rxn.chemutils.tokenization import detokenize_smiles
from rxn.utilities.files import is_path_creatable, load_list_from_file
from rxn.utilities.logging import setup_console_logger

from rxn_onmt_utils.rxn_models.utils import ModelFiles, RxnPreprocessingFiles
from rxn_onmt_utils.translator import Translator


def _standardized(multi_smiles: str) -> ReactionEquation:
    """Helper function putting a prediction into a ReactionEquation and
    standardizing it, in order to enable comparison of prediction and tgt.

    For simplicity, this relies on half-empty reaction equations. The side does
    not actually matter and we put it on the left-hand side also for forward
    prediction.

    Args:
        multi_smiles: product SMILES, or precursors SMILES.
    """
    smiles = detokenize_smiles(multi_smiles)
    reaction = parse_any_reaction_smiles(smiles + ">>")
    return sort_compounds(canonicalize_compounds(reaction))


def _equivalent(pred_smiles: str, tgt_smiles: str) -> bool:
    """Whether the predicted and tgt SMILES are equivalent.

    Works for both forward and retro tasks.
    """

    try:
        return _standardized(pred_smiles) == _standardized(tgt_smiles)
    except Exception:
        return False


@click.command(context_settings=dict(show_default=True))
@click.option(
    "--data_dir",
    type=str,
    required=True,
    help="Directory containing the output of prepare-data",
)
@click.option(
    "--model_dir",
    type=str,
    required=True,
    help="Where to load the model from (will take the last checkpoint)",
)
@click.option("--model_task", type=click.Choice(["forward", "retro"]), required=True)
@click.option(
    "--output_json", type=str, required=True, help="JSON file where to save the metrics"
)
@click.option(
    "--last_checkpoint_filepath",
    type=str,
    required=True,
    help="Filepath where to save the last checkpoint",
)
@click.option("--evaluate_on", type=click.Choice(["valid", "test"]), default="valid")
@click.option("--use_gpu", is_flag=True, help="Run the translation on the GPU")
def evaluate(
    data_dir: str,
    model_dir: str,
    model_task: str,
    output_json: str,
    last_checkpoint_filepath: str,
    evaluate_on: str,
    use_gpu: bool,
) -> None:
    """Evaluate a trained/finetuned OpenNMT model."""

    setup_console_logger()

    # Before translating, verify that we will be able to create the metrics JSON
    if not is_path_creatable(output_json):
        raise RuntimeError(f'"{output_json}" is not writable.')

    model_files = ModelFiles(model_dir)
    data_files = RxnPreprocessingFiles(data_dir)

    last_model = model_files.get_last_checkpoint()
    logging.info(f"Last model checkpoint: {last_model}")

    copy(last_model, last_checkpoint_filepath)
    logging.info(f'Copied "{last_model}" to "{last_checkpoint_filepath}"')

    src = load_list_from_file(
        data_files.get_src_file(split=evaluate_on, model_task=model_task)
    )
    tgt = load_list_from_file(
        data_files.get_tgt_file(split=evaluate_on, model_task=model_task)
    )

    translator = Translator.from_model_path(
        str(last_model), beam_size=10, max_length=300, gpu=(0 if use_gpu else -1)
    )

    logging.info(f"Translating {len(src)} samples...")
    pred = translator.translate_sentences(src)
    logging.info(f"Translating {len(src)} samples... Done.")

    if len(pred) != len(tgt):
        raise RuntimeError(
            f"pred and tgt have different lengths ({len(pred)} != {len(tgt)})"
        )

    logging.info("Canonicalizing and standardizing...")
    n_correct = sum(
        1
        for pred_smiles, tgt_smiles in zip(pred, tgt)
        if _equivalent(pred_smiles, tgt_smiles)
    )
    logging.info("Canonicalizing and standardizing... Done.")

    top1_accuracy = n_correct / len(tgt)
    metrics = {"top1_accuracy": top1_accuracy}
    logging.info(f"Metrics: {metrics}")

    logging.info(f'Saving metrics to "{output_json}"')
    with open(output_json, "wt") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    evaluate()
