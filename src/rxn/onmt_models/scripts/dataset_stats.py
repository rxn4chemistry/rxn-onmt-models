import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
from attr import define
from rxn.chemutils.miscellaneous import get_individual_compounds
from rxn.chemutils.tokenization import detokenize_smiles, file_is_tokenized, to_tokens
from rxn.utilities.files import iterate_lines_from_file
from rxn.utilities.logging import setup_console_logger

from rxn.onmt_models.training_files import RxnPreprocessingFiles

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def number_compounds(smiles_list: List[str]) -> int:
    return len(smiles_list)


def number_characters(smiles_list: List[str]) -> int:
    return len(".".join(smiles_list))


def number_tokens(smiles_list: List[str]) -> Optional[int]:
    try:
        return len(to_tokens(".".join(smiles_list)))
    except Exception:
        return None


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


def get_data_file_path(
    src_or_tgt: str, data_files: RxnPreprocessingFiles, model_task: str, split: str
) -> Path:
    """
    Get the filepath to the file with data (one item per reaction).

    Args:
        src_or_tgt: "src" or "tgt".
        data_files: structure indicating the directory for the data.
        model_task: model task.
        split: "train", "validation", or "test".
    """
    if src_or_tgt == "src":
        return data_files.get_src_file(model_task=model_task, split=split)
    elif src_or_tgt == "tgt":
        return data_files.get_tgt_file(model_task=model_task, split=split)
    else:
        raise ValueError(f'Invalid value: "{src_or_tgt}" (should be "src" or "tgt"')


@define
class NumberStat:
    """Wrapper for a function to obtain a value for a given list of compounds,
    associated with a name.

    An example for the number of compounds could be: NumberStat("n_smiles", len).
    """

    prefix: str
    fn: Callable[[List[str]], Optional[float]]


def make_copy_and_fill(
    results_base: Dict[str, str],
    split: str,
    values_lists: List[List[float]],
    stats: List[NumberStat],
) -> Dict[str, Any]:
    """Extracted to avoid duplication.

    Args:
        results_base: "template" dictionary with global info on dataset and file.
        split: which split is considered.
        values_lists: lists of lists of values to add (one list per stat).
        stats: which stats were generated.
    """
    results_dict = {**results_base, "split": split}
    for values_list, stat in zip(values_lists, stats):
        results_dict.update(
            {
                f"{stat.prefix}_mean": np.mean(values_list),
                f"{stat.prefix}_std": np.std(values_list),
            }
        )
    return results_dict


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

    stats = [
        NumberStat("n_compounds", fn=number_compounds),
        NumberStat("n_chars", fn=number_characters),
        NumberStat("n_tokens", fn=number_tokens),
    ]
    n_stats = len(stats)
    splits = ["train", "validation", "test"]

    results_dicts = []
    for directory in directories:
        logger.info(f'Getting stats for dataset in "{directory}"')
        data_files = RxnPreprocessingFiles(directory)

        for src_or_tgt in ["src", "tgt"]:
            # Create a list of empty lists to contain the aggregated values from all splits
            stats_for_all: List[List[float]] = [[] for _ in range(n_stats)]

            results_base = {
                "dataset_path": str(data_files.processed_data_dir),
                "dataset": data_files.processed_data_dir.name,
                "set": src_or_tgt,
            }

            for split in splits:
                path = get_data_file_path(
                    src_or_tgt=src_or_tgt,
                    data_files=data_files,
                    model_task=model_task,
                    split=split,
                )

                stats_for_split: List[List[float]] = [[] for _ in range(n_stats)]

                # Compute all the stats, one line at a time
                for smiles_group in iter_groups_of_smiles(path):
                    for values_list, stat in zip(stats_for_split, stats):
                        value = stat.fn(smiles_group)
                        if value is not None:
                            values_list.append(value)

                # Add values for the split to the global one
                for values, values_all in zip(stats_for_split, stats_for_all):
                    values_all.extend(values)

                # Add one row for the selected split
                results_dicts.append(
                    make_copy_and_fill(
                        results_base,
                        split=split,
                        values_lists=stats_for_split,
                        stats=stats,
                    )
                )

            # Add one row aggregating all the splits
            results_dicts.append(
                make_copy_and_fill(
                    results_base,
                    split="all",
                    values_lists=stats_for_all,
                    stats=stats,
                )
            )

    df = pd.DataFrame(results_dicts)

    df.to_csv(csv, index=False)
    logger.info(f'Wrote results to "{csv}".')


if __name__ == "__main__":
    main()
