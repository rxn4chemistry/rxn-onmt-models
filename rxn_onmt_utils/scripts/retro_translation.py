import logging
from typing import Optional

import click

from rxn_onmt_utils.rxn_models.forward_or_retro_translation import forward_or_retro_translation

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command()
@click.option('--product_file', required=True, help='File containing the products')
@click.option('--precursors_tgt', type=str, help='Ground truth precursors (optional)')
@click.option(
    '--precursors_pred',
    required=True,
    help='Where to save the predicted precursors (detokenized)'
)
@click.option('--retro_model', required=True, help='Path to the retro model')
@click.option('--n_best', required=True, default=1, type=int, help='Number of predictions to make')
@click.option('--batch_size', required=True, default=64, type=int, help='Batch size')
@click.option('--gpu', is_flag=True, help='If given, run the predictions on a GPU.')
def main(
    product_file: str, precursors_tgt: Optional[str], precursors_pred: str, retro_model: str,
    n_best: int, batch_size: int, gpu: bool
) -> None:
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level='INFO')
    forward_or_retro_translation(
        src_file=product_file,
        tgt_file=precursors_tgt,
        pred_file=precursors_pred,
        model=retro_model,
        n_best=n_best,
        beam_size=15,
        batch_size=batch_size,
        gpu=gpu
    )


if __name__ == '__main__':
    main()
