import logging
from pathlib import Path
from typing import Optional, Union

import click
from rxn.chemutils.tokenization import detokenize_smiles
from rxn.utilities.files import (
    dump_list_to_file,
    iterate_lines_from_file,
    load_list_from_file,
)
from rxn.utilities.logging import setup_console_and_file_logger

from rxn_onmt_utils.rxn_models.classification_translation import (
    classification_translation,
)
from rxn_onmt_utils.rxn_models.forward_or_retro_translation import rxn_translation
from rxn_onmt_utils.rxn_models.metrics_files import RetroFiles
from rxn_onmt_utils.rxn_models.run_metrics import evaluate_metrics
from rxn_onmt_utils.rxn_models.tokenize_file import copy_as_detokenized
from rxn_onmt_utils.rxn_models.utils import (
    convert_class_token_idx_for_tranlation_models,
    raise_if_identical_path,
)
from rxn_onmt_utils.scripts.canonicalize_file import canonicalize_file
from rxn_onmt_utils.utils import ensure_directory_exists_and_is_empty

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def create_rxn_from_files(
    input_file_precursors: Union[str, Path],
    input_file_products: Union[str, Path],
    output_file: Union[str, Path],
) -> None:
    raise_if_identical_path(input_file_precursors, output_file)
    raise_if_identical_path(input_file_products, output_file)
    logger.info(
        f'Combining files "{input_file_precursors}" and "{input_file_products}" -> "{output_file}".'
    )

    precursors = load_list_from_file(input_file_precursors)
    products = load_list_from_file(input_file_products)

    rxn = [f"{prec}>>{prod}" for prec, prod in zip(precursors, products)]
    dump_list_to_file(rxn, output_file)


@click.command(context_settings={"show_default": True})
@click.option(
    "--precursors_file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="File containing the precursors of a test set",
)
@click.option(
    "--products_file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="File containing the products of a test set",
)
@click.option(
    "--output_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Where to save all the files",
)
@click.option(
    "--retro_model",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the single-step retrosynthesis model",
)
@click.option(
    "--forward_model",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the forward model",
)
@click.option(
    "--classification_model",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    default=None,
    help="Path to the classification model",
)
@click.option("--batch_size", default=64, type=int, help="Batch size")
@click.option(
    "--n_best", default=10, type=int, help="Number of retro predictions to make (top-N)"
)
@click.option("--gpu", is_flag=True, help="If given, run the predictions on a GPU.")
@click.option(
    "--no_metrics", is_flag=True, help="If given, the metrics will not be computed."
)
@click.option(
    "--beam_size", default=15, type=int, help="Beam size for retro (> n_best)."
)
@click.option(
    "--class_tokens",
    default=None,
    type=int,
    help="The number of tokens used in the trainings",
)
def main(
    precursors_file: Path,
    products_file: Path,
    output_dir: Path,
    retro_model: Path,
    forward_model: Path,
    classification_model: Optional[Path],
    batch_size: int,
    n_best: int,
    gpu: bool,
    no_metrics: bool,
    beam_size: int,
    class_tokens: Optional[int],
) -> None:
    """Starting from the ground truth files and two models (retro, forward),
    generate the translation files needed for the metrics, and calculate the default metrics.
    """

    ensure_directory_exists_and_is_empty(output_dir)
    retro_files = RetroFiles(output_dir)

    setup_console_and_file_logger(retro_files.log_file)

    if class_tokens is not None:
        class_token_products = (
            f"{convert_class_token_idx_for_tranlation_models(class_token_idx)}{detokenize_smiles(line)}"
            for line in iterate_lines_from_file(products_file)
            for class_token_idx in range(class_tokens)
        )
        class_token_precursors = (
            detokenize_smiles(line)
            for line in iterate_lines_from_file(precursors_file)
            for _ in range(class_tokens)
        )
        dump_list_to_file(class_token_products, retro_files.class_token_products)
        dump_list_to_file(class_token_precursors, retro_files.class_token_precursors)

    copy_as_detokenized(products_file, retro_files.gt_src)
    copy_as_detokenized(precursors_file, retro_files.gt_tgt)

    # retro
    rxn_translation(
        src_file=(
            retro_files.gt_src
            if class_tokens is None
            else retro_files.class_token_products
        ),
        tgt_file=(
            retro_files.gt_tgt
            if class_tokens is None
            else retro_files.class_token_precursors
        ),
        pred_file=retro_files.predicted,
        model=retro_model,
        n_best=n_best,
        beam_size=beam_size,
        batch_size=batch_size,
        gpu=gpu,
    )

    canonicalize_file(
        retro_files.predicted,
        retro_files.predicted_canonical,
        invalid_placeholder="",
        sort_molecules=True,
    )

    # Forward
    rxn_translation(
        src_file=retro_files.predicted_canonical,
        tgt_file=None,
        pred_file=retro_files.predicted_products,
        model=forward_model,
        n_best=1,
        beam_size=10,
        batch_size=batch_size,
        gpu=gpu,
    )

    canonicalize_file(
        retro_files.predicted_products,
        retro_files.predicted_products_canonical,
        invalid_placeholder="",
    )

    if classification_model:
        create_rxn_from_files(
            retro_files.predicted_canonical,
            retro_files.predicted_products_canonical,
            retro_files.predicted_rxn_canonical,
        )

        # Classification
        classification_translation(
            src_file=retro_files.predicted_rxn_canonical,
            tgt_file=None,
            pred_file=retro_files.predicted_classes,
            model=classification_model,
            n_best=1,
            beam_size=5,
            batch_size=batch_size,
            gpu=gpu,
        )

    if not no_metrics:
        evaluate_metrics("retro", output_dir)


if __name__ == "__main__":
    main()
