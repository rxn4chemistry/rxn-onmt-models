import logging
import random
from pathlib import Path
from typing import Tuple

import click
from rxn.utilities.logging import setup_console_logger

from rxn_onmt_utils.rxn_models.augmentation import augment_translation_dataset
from rxn_onmt_utils.rxn_models.utils import RxnPreprocessingFiles

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command(context_settings=dict(show_default=True))
@click.option(
    "--data_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to the output directory of rxn-prepare-data.",
)
@click.option(
    "--model_task", type=click.Choice(["forward", "retro", "context"]), required=True
)
@click.option(
    "--splits",
    "-s",
    type=click.Choice(["train", "validation", "test"]),
    default=("train",),
    multiple=True,
    help="Which split(s) to augment.",
)
@click.option(
    "--n_augmentations",
    "-n",
    type=int,
    required=True,
    help="How many augmented samples to produce for each input.",
)
@click.option(
    "--keep_original/--discard_original",
    default=True,
    help="Whether to keep the original sample along the augmented ones.",
)
@click.option(
    "--seed",
    default=42,
    help="Random seed.",
)
def main(
    data_dir: Path,
    model_task: str,
    splits: Tuple[str, ...],
    n_augmentations: int,
    keep_original: bool,
    seed: int,
) -> None:
    """
    Augment the training data.

    Notes:
        1) the input is augmented, while the output is not. But it may need
           to be duplicated if there are several augmented samples for each input.
        2) Preferred to the augmentation in rxn-reaction-preprocessing, which
           is limited to forward and retro predictions.
    """
    setup_console_logger()

    random.seed(seed)

    data_files = RxnPreprocessingFiles(data_dir)

    for split in splits:
        logger.info(f"Augmenting split: {split}")

        src = data_files.get_src_file(split, model_task)
        tgt = data_files.get_tgt_file(split, model_task)

        src_augmented = data_files.augmented(src)
        tgt_augmented = data_files.augmented(tgt)

        logger.info(
            f'Augmenting the dataset: "{src}" -> "{src_augmented}" and '
            f'"{tgt}" -> {tgt_augmented}"'
        )
        augment_translation_dataset(
            src_in=src,
            src_out=src_augmented,
            tgt_in=tgt,
            tgt_out=tgt_augmented,
            n_augmentations=n_augmentations,
            keep_original=keep_original,
        )


if __name__ == "__main__":
    main()
