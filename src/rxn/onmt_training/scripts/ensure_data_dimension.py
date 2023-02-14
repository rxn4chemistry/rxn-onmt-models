import logging
import math
from pathlib import Path
from typing import Iterable, Tuple

import click
from rxn.utilities.files import (
    PathLike,
    count_lines,
    dump_list_to_file,
    load_list_from_file,
)
from rxn.utilities.logging import setup_console_logger

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def ensure_data_dimension(
    txt_files: Iterable[PathLike], output_dir: PathLike, max_dimension: int
) -> None:
    # Check the lengths of the files and ensure they are all the same
    file_length = [count_lines(txt_file) for txt_file in txt_files]
    if len(set(file_length)) != 1:
        raise ValueError("The files provided have not the same number of lines.")

    # Check that there are no files with the same name
    filenames = [Path(txt_file).name for txt_file in txt_files]
    if len(set(filenames)) != len(filenames):
        raise ValueError("Found files with the same same. Aborting")

    split_no = math.ceil(file_length[0] / max_dimension)

    new_output_dir = Path(output_dir)
    new_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Splitting in {split_no} files with the same name of the original ones . "
        f"Saving in {new_output_dir} ."
    )

    for txt_file in txt_files:
        file_content = load_list_from_file(txt_file)
        file_name = Path(txt_file).name
        for chunk_no, chunk_start in enumerate(range(0, file_length[0], max_dimension)):
            # create a sub_directory
            sub_directory = Path(new_output_dir) / f"chunk_{chunk_no}"
            sub_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory {sub_directory} . Saving files .")

            # save all subfiles
            dump_list_to_file(
                file_content[chunk_start : chunk_start + max_dimension],
                Path(sub_directory) / file_name,
            )


@click.command(context_settings={"show_default": True})
@click.argument("txt_files", nargs=-1)
@click.option("--output_dir", required=True, help="Where to save all the files")
@click.option(
    "--max_dimension",
    default=50000,
    type=int,
    help="Maximum file length allowed without splitting",
)
def main(txt_files: Tuple[str, ...], output_dir: str, max_dimension: int) -> None:
    """
    Script to split too big files in subchunks . Useful for class token translations.
    Takes as input an arbitrary number of files. Files are saved under output_dir/chunk_i
    for i ranging from 0 to the number of splits needed.
    """
    setup_console_logger()

    ensure_data_dimension(txt_files, output_dir, max_dimension)


if __name__ == "__main__":
    main()
