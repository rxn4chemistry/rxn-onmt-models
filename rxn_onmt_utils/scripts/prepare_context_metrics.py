import json
import logging
from pathlib import Path

import click
from rxn.utilities.logging import setup_console_and_file_logger

from rxn_onmt_utils.rxn_models.context_metrics import ContextMetrics
from rxn_onmt_utils.rxn_models.forward_or_retro_translation import (
    forward_or_retro_translation,
)
from rxn_onmt_utils.rxn_models.tokenize_file import copy_as_detokenized
from rxn_onmt_utils.rxn_models.utils import ContextFiles
from rxn_onmt_utils.scripts.canonicalize_file import canonicalize_file
from rxn_onmt_utils.utils import ensure_directory_exists_and_is_empty

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command(context_settings={"show_default": True})
@click.option(
    "--src_file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="File containing the src of the context prediction task",
)
@click.option(
    "--tgt_file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="File containing the tgt of the context prediction task",
)
@click.option(
    "--output_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Where to save all the files",
)
@click.option(
    "--context_model",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the context model",
)
@click.option("--batch_size", default=64, type=int, help="Batch size")
@click.option(
    "--n_best", default=5, type=int, help="Number of retro predictions to make (top-N)"
)
@click.option("--gpu", is_flag=True, help="If given, run the predictions on a GPU.")
@click.option(
    "--no_metrics", is_flag=True, help="If given, the metrics will not be computed."
)
def main(
    src_file: Path,
    tgt_file: Path,
    output_dir: Path,
    context_model: Path,
    batch_size: int,
    n_best: int,
    gpu: bool,
    no_metrics: bool,
) -> None:
    """Starting from the ground truth files and context model, generate the
    translation files needed for the metrics, and calculate the default metrics."""

    ensure_directory_exists_and_is_empty(output_dir)
    context_files = ContextFiles(output_dir)

    setup_console_and_file_logger(context_files.log_file)

    copy_as_detokenized(src_file, context_files.gt_src)
    copy_as_detokenized(tgt_file, context_files.gt_tgt)

    # context prediction
    forward_or_retro_translation(
        src_file=context_files.gt_src,
        tgt_file=context_files.gt_tgt,
        pred_file=context_files.predicted_context,
        model=context_model,
        n_best=n_best,
        beam_size=10,
        batch_size=batch_size,
        gpu=gpu,
    )

    canonicalize_file(
        context_files.predicted_context,
        context_files.predicted_context_canonical,
        invalid_placeholder="",
    )

    if not no_metrics:
        logger.info("Computing the context metrics...")
        metrics = ContextMetrics.from_context_files(context_files)
        metrics_dict = metrics.get_metrics()
        with open(context_files.metrics_file, "wt") as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(
            f'Computing the context metrics... Saved to "{context_files.metrics_file}".'
        )


if __name__ == "__main__":
    main()
