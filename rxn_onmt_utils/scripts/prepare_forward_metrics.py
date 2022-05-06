import json
import logging
from pathlib import Path

import click

from rxn_onmt_utils.rxn_models.forward_metrics import ForwardMetrics
from rxn_onmt_utils.rxn_models.forward_or_retro_translation import forward_or_retro_translation
from rxn_onmt_utils.rxn_models.tokenize_file import copy_as_detokenized
from rxn_onmt_utils.rxn_models.utils import ForwardFiles
from rxn_onmt_utils.scripts.canonicalize_file import canonicalize_file

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command(context_settings={'show_default': True})
@click.option(
    '--precursors_file', required=True, help='File containing the precursors of a test set'
)
@click.option('--products_file', required=True, help='File containing the products of a test set')
@click.option('--output_dir', required=True, help='Where to save all the files')
@click.option('--forward_model', required=True, help='Path to the forward model')
@click.option('--batch_size', default=64, type=int, help='Batch size')
@click.option('--n_best', default=5, type=int, help='Number of retro predictions to make (top-N)')
@click.option('--gpu', is_flag=True, help='If given, run the predictions on a GPU.')
@click.option('--no_metrics', is_flag=True, help='If given, the metrics will not be computed.')
def main(
    precursors_file: str, products_file: str, output_dir: str, forward_model: str, batch_size: int,
    n_best: int, gpu: bool, no_metrics: bool
) -> None:
    """Starting from the ground truth files and forward model, generate the
    translation files needed for the metrics, and calculate the default metrics."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path_contains_files = any(output_path.iterdir())
    if output_path_contains_files:
        raise RuntimeError(f'The output directory "{output_path}" is required to be empty.')

    forward_files = ForwardFiles(output_path)

    # Setup logging (to terminal and to file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(forward_files.log_file),
                  logging.StreamHandler()]
    )

    copy_as_detokenized(products_file, forward_files.gt_products)
    copy_as_detokenized(precursors_file, forward_files.gt_precursors)

    # Forward
    forward_or_retro_translation(
        src_file=forward_files.gt_precursors,
        tgt_file=forward_files.gt_products,
        pred_file=forward_files.predicted_products,
        model=forward_model,
        n_best=n_best,
        beam_size=10,
        batch_size=batch_size,
        gpu=gpu
    )

    canonicalize_file(
        forward_files.predicted_products,
        forward_files.predicted_products_canonical,
        invalid_placeholder=''
    )

    if not no_metrics:
        logger.info('Computing the forward metrics...')
        metrics = ForwardMetrics.from_forward_files(forward_files)
        metrics_dict = metrics.get_metrics()
        with open(forward_files.metrics_file, 'wt') as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f'Computing the forward metrics... Saved to "{forward_files.metrics_file}".')


if __name__ == '__main__':
    main()
