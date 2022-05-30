import logging
from pathlib import Path
from typing import Optional

import click
from rxn_utilities.logging_utilities import setup_console_logger

from rxn_onmt_utils import __version__
from rxn_onmt_utils.rxn_models.forward_or_retro_translation import forward_or_retro_translation

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command(context_settings={'show_default': True})
@click.option(
    '--model',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Path to the RXN model (forward, retro, etc.)'
)
@click.option(
    '--src_file',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Path to the translation input'
)
@click.option(
    '--pred_file',
    required=True,
    type=click.Path(writable=True, path_type=Path),
    help='Path where to save the predictions'
)
@click.option(
    '--tgt_file',
    type=click.Path(exists=True, path_type=Path),
    help='Optional, path to the ground truth'
)
@click.option('--batch_size', default=64, type=int, help='Batch size')
@click.option('--n_best', default=1, type=int, help='Number of retro predictions to make (top-N)')
@click.option('--beam_size', default=10, type=int, help='Beam size for retro (> n_best).')
@click.option('--no_gpu', is_flag=True, help='Run the training on CPU (slow!)')
def main(
    model: Path,
    src_file: Path,
    pred_file: Path,
    tgt_file: Optional[Path],
    batch_size: int,
    n_best: int,
    beam_size: int,
    no_gpu: bool,
) -> None:
    """Translate with an RXN model."""

    setup_console_logger()

    logger.info(
        f'RXN translation "{src_file}" -> "{tgt_file}" with model "{model}". '
        'Note: there is no post-processing of the predictions.'
    )
    logger.info(f'rxn-onmt-utils version: {__version__}. ')

    forward_or_retro_translation(
        src_file=src_file,
        tgt_file=tgt_file,
        pred_file=pred_file,
        model=model,
        n_best=n_best,
        beam_size=beam_size,
        batch_size=batch_size,
        gpu=not no_gpu
    )


if __name__ == '__main__':
    main()
