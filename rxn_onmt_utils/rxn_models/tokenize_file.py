import logging
import shutil

from rxn.chemutils.tokenization import (
    TokenizationError,
    detokenize_smiles,
    tokenize_smiles,
)
from rxn.utilities.files import PathLike, dump_list_to_file, iterate_lines_from_file

from rxn_onmt_utils.rxn_models.utils import (
    UnclearWhetherTokenized,
    raise_if_identical_path,
    string_is_tokenized,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def tokenize_line(smiles_line: str, invalid_placeholder: str) -> str:
    try:
        return tokenize_smiles(smiles_line)
    except TokenizationError:
        logger.debug(f'Error when tokenizing "{smiles_line}"')
        return invalid_placeholder


def tokenize_file(
    input_file: PathLike, output_file: PathLike, invalid_placeholder: str = ""
) -> None:
    raise_if_identical_path(input_file, output_file)
    logger.info(f'Tokenizing "{input_file}" -> "{output_file}".')

    tokenized = (
        tokenize_line(line, invalid_placeholder)
        for line in iterate_lines_from_file(input_file)
    )

    dump_list_to_file(tokenized, output_file)


def detokenize_file(
    input_file: PathLike,
    output_file: PathLike,
) -> None:
    raise_if_identical_path(input_file, output_file)
    logger.info(f'Detokenizing "{input_file}" -> "{output_file}".')

    detokenized = (
        detokenize_smiles(line) for line in iterate_lines_from_file(input_file)
    )
    dump_list_to_file(detokenized, output_file)


def ensure_tokenized_file(
    file: PathLike, postfix: str = ".tokenized", invalid_placeholder: str = ""
) -> str:
    """
    Ensure that a file is tokenized: do nothing if the file is already tokenized, create
    a tokenized copy otherwise.

    Args:
        file: path to the file that we want to ensure is tokenized.
        postfix: postfix to add to the tokenized copy (if applicable).
        invalid_placeholder: placeholder for lines that cannot be tokenized (if applicable).

    Returns:
        The path to the tokenized file (original path, or path to new file).
    """
    if file_is_tokenized(file):
        return str(file)

    tokenized_copy = str(file) + postfix
    tokenize_file(file, tokenized_copy, invalid_placeholder=invalid_placeholder)
    return tokenized_copy


def detokenize_class(tokenized_class: str) -> str:
    """
    Function performing a detokenization of the reaction class used in the Transformer classification
    model. E.g. '1 1.2 1.2.3' -> 1.2.3

    Args:
        tokenized_class: str to detokenize

    Raises:
        ValueError: if the input string format is not correct
    """
    if tokenized_class == "0":
        return tokenized_class

    splitted_class = tokenized_class.split(" ")
    if len(splitted_class) == 1 and len(splitted_class[0].split(".")) == 3:
        # here the class is already detokenized
        return tokenized_class
    if len(splitted_class) != 3:
        raise ValueError(
            f'The class to be detokenized, "{tokenized_class}", is probably not in the correct format.'
        )
    return splitted_class[-1]


def tokenize_class(detokenized_class: str) -> str:
    """
    Function performing a tokenization of the reaction class used in the Transformer classification
    model. E.g. '1.2.3' -> '1 1.2 1.2.3'

    Args:
        detokenized_class: str to tokenize

    Raises:
        ValueError: if the input string format is not correct
    """
    if detokenized_class == "0":
        return detokenized_class

    splitted_class = detokenized_class.split(".")
    if len(splitted_class) == 4 and len(detokenized_class.split(" ")) == 3:
        # here the class is already tokenized
        return detokenized_class
    if len(splitted_class) != 3:
        raise ValueError(
            f'The class to be tokenized, "{detokenized_class}", is probably not in the correct format.'
        )
    a, b, _ = splitted_class
    return f"{a} {a}.{b} {detokenized_class}"


def tokenize_class_line(class_line: str, invalid_placeholder: str) -> str:
    try:
        return tokenize_class(class_line)
    except ValueError:
        logger.debug(f'Error when tokenizing the class "{class_line}"')
        return invalid_placeholder


def detokenize_class_line(class_line: str, invalid_placeholder: str) -> str:
    try:
        return detokenize_class(class_line)
    except ValueError:
        logger.debug(f'Error when detokenizing the class "{class_line}"')
        return invalid_placeholder


def detokenize_classification_file(
    input_file: PathLike, output_file: PathLike, invalid_placeholder: str = ""
) -> None:
    raise_if_identical_path(input_file, output_file)
    logger.info(f'Detokenizing "{input_file}" -> "{output_file}".')

    detokenized = (
        detokenize_class_line(line, invalid_placeholder)
        for line in iterate_lines_from_file(input_file)
    )
    dump_list_to_file(detokenized, output_file)


def tokenize_classification_file(
    input_file: PathLike, output_file: PathLike, invalid_placeholder: str = ""
) -> None:
    raise_if_identical_path(input_file, output_file)
    logger.info(f'Tokenizing "{input_file}" -> "{output_file}".')

    tokenized = (
        tokenize_class_line(line, invalid_placeholder)
        for line in iterate_lines_from_file(input_file)
    )
    dump_list_to_file(tokenized, output_file)


def classification_string_is_tokenized(classification_line: str) -> bool:
    """
    Whether a classification line is tokenized or not.

    Args:
        classification_line: line to inspect

    Raises:
        ValueError: for errors in tokenization or detokenization
    """
    detokenized = detokenize_class(classification_line)
    tokenized = tokenize_class(detokenized)
    return classification_line == tokenized


def classification_file_is_tokenized(filepath: PathLike) -> bool:
    """
    Whether a file contains tokenized classes or not.
    '1.2.3' -> '1 1.2 1.2.3'

    By default, this looks at the first non-empty line of the file only!

    Raises:
        ValueError: for errors in tokenization or detokenization
        RuntimeError: for empty files or files with empty lines only.

    Args:
        filepath: path to the file.
    """
    for line in iterate_lines_from_file(filepath):
        # Ignore empty lines
        if line == "":
            continue
        return classification_string_is_tokenized(line)
    raise RuntimeError(
        f'Could not determine whether "{filepath}" is class-tokenized: empty lines only.'
    )


def file_is_tokenized(filepath: PathLike) -> bool:
    """
    Whether a file contains tokenized SMILES or not.

    By default, this looks at the first non-empty line of the file only!

    Raises:
        TokenizationError: propagated from tokenize_smiles()
        RuntimeError: for empty files or files with empty lines only.

    Args:
        filepath: path to the file.
    """
    # Iterative formulation in case the first line(s) of the file don't make it
    # clear whether tokenized or not.
    for line in iterate_lines_from_file(filepath):
        try:
            return string_is_tokenized(line)
        except UnclearWhetherTokenized:
            continue
    raise RuntimeError(
        f'Could not determine whether "{filepath}" is tokenized: empty lines only.'
    )


def copy_as_detokenized(src: PathLike, dest: PathLike) -> None:
    """
    Copy a source file to a destination, while making sure that it is not tokenized.
    """
    if file_is_tokenized(src):
        logger.info(f'Copying and detokenizing "{src}" -> "{dest}".')
        detokenize_file(src, dest)
    else:
        logger.info(f'Copying "{src}" -> "{dest}".')
        shutil.copy(src, dest)
