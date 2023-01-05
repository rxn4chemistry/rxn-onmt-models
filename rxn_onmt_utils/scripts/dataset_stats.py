import logging
from pathlib import Path
from typing import Iterator, List, Tuple

import click
import numpy as np
import pandas as pd
from rxn.chemutils.miscellaneous import get_individual_compounds
from rxn.chemutils.tokenization import detokenize_smiles
from rxn.utilities.files import iterate_lines_from_file
from rxn.utilities.logging import setup_console_logger

from rxn_onmt_utils.rxn_models.tokenize_file import file_is_tokenized
from rxn_onmt_utils.rxn_models.utils import RxnPreprocessingFiles

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def iter_groups_of_smiles(path: Path) -> Iterator[List[str]]:
    """Get the individual compounds in SMILES format for each line of a file."""
    is_tokenized = file_is_tokenized(path)

    for line in iterate_lines_from_file(path):
        if is_tokenized:
            line = detokenize_smiles(line)
        try:
            yield get_individual_compounds(line)
        except Exception as e:
            logger.error(f'Could not get the compounds for "{line}": {e}')


@click.command()
@click.option("--csv", required=True, help="Where to save the csv")
@click.option(
    "--model_task", type=click.Choice(["forward", "retro", "context"]), required=True
)
@click.argument("directories", nargs=-1)
def main(csv: str, model_task: str, directories: Tuple[str, ...]) -> None:
    """Get the stats for one or several datasets produced by rxn-prepare-data and
    collect them into a CSV.

    Usage examples:
        - rxn-dataset-stats --csv stats.csv dir1 dir2 dir3
        - rxn-dataset-stats --csv stats.csv dir* other_dir
        - rxn-dataset-stats --csv stats.csv *
    """
    setup_console_logger()

    splits = ["train", "validation", "test"]

    results_dicts = []
    for directory in directories:
        logger.info(f'Getting stats for dataset in "{directory}"')
        data_files = RxnPreprocessingFiles(directory)

        for split in splits:

            src = data_files.get_src_file(split, model_task)
            tgt = data_files.get_tgt_file(split, model_task)

            for tag, path in [("src", src), ("tgt", tgt)]:
                list_of_n_compounds = []
                list_of_n_chars = []
                for smiles_group in iter_groups_of_smiles(path):
                    list_of_n_compounds.append(len(smiles_group))
                    list_of_n_chars.append(
                        len(".".join(smiles_group))
                    )  # Note: this is approximate

                results_dicts.append(
                    {
                        "dataset_path": str(data_files.processed_data_dir),
                        "dataset": data_files.processed_data_dir.name,
                        "set": tag,
                        "split": split,
                        "n_compounds_mean": np.mean(list_of_n_compounds),
                        "n_compounds_std": np.std(list_of_n_compounds),
                        "n_chars_mean": np.mean(list_of_n_chars),
                        "n_chars_std": np.std(list_of_n_chars),
                    }
                )

    df = pd.DataFrame(results_dicts)

    df.to_csv(csv, index=False)
    logger.info(f'Wrote results to "{csv}".')


if __name__ == "__main__":
    main()
