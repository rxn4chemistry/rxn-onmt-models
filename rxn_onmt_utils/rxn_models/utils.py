import logging
import os
from pathlib import Path
from typing import List, Sequence

from rxn_chemutils.tokenization import detokenize_smiles, tokenize_smiles
from rxn_utilities.file_utilities import iterate_lines_from_file, PathLike

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def raise_if_identical_path(input_path: PathLike, output_path: PathLike) -> None:
    """
    Raise an exception if input and output paths point to the same file.
    """
    if os.path.realpath(input_path) == os.path.realpath(output_path):
        raise ValueError(
            f'The output path, "{output_path}", must be '
            f'different from the input path, "{input_path}".'
        )


def string_is_tokenized(smiles_line: str) -> bool:
    """
    Whether a line is tokenized or not.

    Args:
        smiles_line: line to inspect

    Raises:
        TokenizationError: propagated directly from tokenize_smiles()
    """
    detokenized = detokenize_smiles(smiles_line)
    tokenized = tokenize_smiles(detokenized)
    return smiles_line == tokenized


def file_is_tokenized(filepath: PathLike) -> bool:
    """
    Whether a file contains tokenized SMILES or not.

    By default, this looks at the first line of the file only!

    Raises:
        TokenizationError: propagated from tokenize_smiles()
        StopIteration: for empty files

    Args:
        filepath: path to the file.
    """
    first_line = next(iterate_lines_from_file(filepath))
    return string_is_tokenized(first_line)


class MetricsFiles:

    def __init__(self, directory: PathLike):
        self.directory = Path(directory)
        self.log_file = self.directory / 'log.txt'
        self.metrics_file = self.directory / 'metrics.json'


class RetroFiles(MetricsFiles):
    """
    Class holding the locations of the files to write to or to read from for
    the evaluation of retro metrics.
    """

    def __init__(self, directory: PathLike):
        super().__init__(directory=directory)
        self.gt_products = self.directory / 'gt_products.txt'
        self.gt_precursors = self.directory / 'gt_precursors.txt'
        self.predicted_precursors = self.directory / 'predicted_precursors.txt'
        self.predicted_precursors_canonical = self.directory / 'predicted_precursors_canonical.txt'
        self.predicted_products = self.directory / 'predicted_products.txt'
        self.predicted_products_canonical = self.directory / 'predicted_products_canonical.txt'


class ForwardFiles(MetricsFiles):
    """
    Class holding the locations of the files to write to or to read from for
    the evaluation of forward metrics.
    """

    def __init__(self, directory: PathLike):
        super().__init__(directory=directory)
        self.gt_products = self.directory / 'gt_products.txt'
        self.gt_precursors = self.directory / 'gt_precursors.txt'
        self.predicted_products = self.directory / 'predicted_products.txt'
        self.predicted_products_canonical = self.directory / 'predicted_products_canonical.txt'


def preprocessed_id_names(n_additional_sets: int) -> List[str]:
    """Get the names of the ids for the datasets used in multi-task training
    with OpenNMT.

    Args:
        n_additional_sets: how many sets there are in addition to the main set.
    """
    return ['main_set'] + [f'additional_set_{i+1}' for i in range(n_additional_sets)]


def extend_command_args_for_gpu(command_and_args: List[str], *, no_gpu: bool) -> None:
    """
    Extend the command with what is needed for execution on GPU.

    `no_gpu` is given as a keyword-only argument to avoid confusion.
    """
    if not no_gpu:
        command_and_args.extend(['-gpu_ranks', '0'])


def extend_command_args_for_data_weights(
    command_and_args: List[str], data_weights: Sequence[int]
) -> None:
    if data_weights:
        n_additional_datasets = len(data_weights) - 1
        data_ids = preprocessed_id_names(n_additional_datasets)
        command_and_args.extend(
            [
                '-data_ids',
                *data_ids,
                '-data_weights',
                *(str(weight) for weight in data_weights),
            ]
        )
