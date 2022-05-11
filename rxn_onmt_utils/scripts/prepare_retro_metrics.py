import json
import logging
from pathlib import Path

import click

from rxn_onmt_utils.rxn_models.forward_or_retro_translation import forward_or_retro_translation
from rxn_onmt_utils.rxn_models.retro_metrics import RetroMetrics
from rxn_onmt_utils.rxn_models.tokenize_file import copy_as_detokenized
from rxn_onmt_utils.rxn_models.utils import RetroFiles
from rxn_onmt_utils.scripts.canonicalize_file import canonicalize_file
from rxn_onmt_utils.utils import setup_console_and_file_logger

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command(context_settings={'show_default': True})
@click.option(
    '--precursors_file', required=True, help='File containing the precursors of a test set'
)
@click.option('--products_file', required=True, help='File containing the products of a test set')
@click.option('--output_dir', required=True, help='Where to save all the files')
@click.option('--retro_model', required=True, help='Path to the single-step retrosynthesis model')
@click.option('--forward_model', required=True, help='Path to the forward model')
@click.option('--batch_size', default=64, type=int, help='Batch size')
@click.option('--n_best', default=10, type=int, help='Number of retro predictions to make (top-N)')
@click.option('--gpu', is_flag=True, help='If given, run the predictions on a GPU.')
@click.option('--no_metrics', is_flag=True, help='If given, the metrics will not be computed.')
def main(
    precursors_file: str, products_file: str, output_dir: str, retro_model: str,
    forward_model: str, batch_size: int, n_best: int, gpu: bool, no_metrics: bool
) -> None:
    """Starting from the ground truth files and two models (retro, forward),
    generate the translation files needed for the metrics, and calculate the default metrics."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path_contains_files = any(output_path.iterdir())
    if output_path_contains_files:
        raise RuntimeError(f'The output directory "{output_path}" is required to be empty.')

    retro_files = RetroFiles(output_path)

    # Setup logging (to terminal and to file)
    setup_console_and_file_logger(retro_files.log_file)

    copy_as_detokenized(products_file, retro_files.gt_products)
    copy_as_detokenized(precursors_file, retro_files.gt_precursors)

    # retro
    forward_or_retro_translation(
        src_file=retro_files.gt_products,
        tgt_file=retro_files.gt_precursors,
        pred_file=retro_files.predicted_precursors,
        model=retro_model,
        n_best=n_best,
        beam_size=15,
        batch_size=batch_size,
        gpu=gpu
    )

    canonicalize_file(
        retro_files.predicted_precursors,
        retro_files.predicted_precursors_canonical,
        invalid_placeholder='',
        sort_molecules=True
    )

    # Forward
    forward_or_retro_translation(
        src_file=retro_files.predicted_precursors_canonical,
        tgt_file=None,
        pred_file=retro_files.predicted_products,
        model=forward_model,
        n_best=1,
        beam_size=10,
        batch_size=batch_size,
        gpu=gpu
    )

    canonicalize_file(
        retro_files.predicted_products,
        retro_files.predicted_products_canonical,
        invalid_placeholder=''
    )

    if not no_metrics:
        logger.info('Computing the retro metrics...')
        metrics = RetroMetrics.from_retro_files(retro_files)
        metrics_dict = metrics.get_metrics()
        with open(retro_files.metrics_file, 'wt') as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f'Computing the retro metrics... Saved to "{retro_files.metrics_file}".')


if __name__ == '__main__':
    main()
