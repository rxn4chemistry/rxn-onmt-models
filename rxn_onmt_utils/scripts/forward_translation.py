import logging
from typing import Optional

import click
from rxn.utilities.logging import setup_console_logger

from rxn_onmt_utils.rxn_models.forward_or_retro_translation import (
    forward_or_retro_translation,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command()
@click.option("--precursors_file", required=True, help="File containing the precursors")
@click.option("--products_tgt", type=str, help="Ground truth products (optional)")
@click.option(
    "--products_pred",
    required=True,
    help="Where to save the predicted products (detokenized)",
)
@click.option("--forward_model", required=True, help="Path to the forward model")
@click.option(
    "--n_best", required=True, default=1, type=int, help="Number of predictions to make"
)
@click.option("--batch_size", required=True, default=64, type=int, help="Batch size")
@click.option("--gpu", is_flag=True, help="If given, run the predictions on a GPU.")
def main(
    precursors_file: str,
    products_tgt: Optional[str],
    products_pred: str,
    forward_model: str,
    n_best: int,
    batch_size: int,
    gpu: bool,
) -> None:
    setup_console_logger()
    forward_or_retro_translation(
        src_file=precursors_file,
        tgt_file=products_tgt,
        pred_file=products_pred,
        model=forward_model,
        n_best=n_best,
        beam_size=10,
        batch_size=batch_size,
        gpu=gpu,
    )


if __name__ == "__main__":
    main()
