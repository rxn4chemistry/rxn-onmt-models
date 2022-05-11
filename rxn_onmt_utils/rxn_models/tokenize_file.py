import logging
import shutil

from rxn_chemutils.tokenization import tokenize_smiles, TokenizationError, detokenize_smiles
from rxn_utilities.file_utilities import iterate_lines_from_file, dump_list_to_file, PathLike

from .utils import raise_if_identical_path, file_is_tokenized

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def tokenize_line(smiles_line: str, invalid_placeholder: str) -> str:
    try:
        return tokenize_smiles(smiles_line)
    except TokenizationError:
        logger.debug(f'Error when tokenizing "{smiles_line}"')
        return invalid_placeholder


def tokenize_file(
    input_file: PathLike, output_file: PathLike, invalid_placeholder: str = ''
) -> None:
    raise_if_identical_path(input_file, output_file)
    logger.info(f'Tokenizing "{input_file}" -> "{output_file}".')

    tokenized = (
        tokenize_line(line, invalid_placeholder) for line in iterate_lines_from_file(input_file)
    )

    dump_list_to_file(tokenized, output_file)


def detokenize_file(
    input_file: PathLike,
    output_file: PathLike,
) -> None:
    raise_if_identical_path(input_file, output_file)
    logger.info(f'Detokenizing "{input_file}" -> "{output_file}".')

    detokenized = (detokenize_smiles(line) for line in iterate_lines_from_file(input_file))
    dump_list_to_file(detokenized, output_file)


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
