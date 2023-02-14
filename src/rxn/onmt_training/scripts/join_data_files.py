import logging
import re
import shutil
from pathlib import Path
from typing import List, Tuple

import click
from rxn.utilities.files import PathLike, raise_if_paths_are_identical
from rxn.utilities.logging import setup_console_logger

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def sorted_chunk_directories(input_path: Path) -> List[Path]:
    # We match the directories ending with a number
    directory_and_directory_no: List[Tuple[Path, int]] = []
    for subdir in input_path.iterdir():
        match = re.match(r".*?(\d+)$", str(subdir))
        if match is not None:
            directory_and_directory_no.append((subdir, int(match.group(1))))

    return [
        chunk_directory[0]
        for chunk_directory in sorted(directory_and_directory_no, key=lambda x: x[1])
    ]


def join_data_files(input_dir: PathLike, output_dir: PathLike) -> None:
    """
    Joining files with `shutil`, reference: https://stackoverflow.com/a/27077437
    """
    raise_if_paths_are_identical(input_dir, output_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Assuming that all directories contain the same files
    filenames = [filename.name for filename in (Path(input_dir) / "chunk_0").iterdir()]
    sorted_chunk_dirs = sorted_chunk_directories(Path(input_dir))
    for filename in filenames:
        out_file_path = output_path / filename
        logger.info(f"Joining files of type: {filename}")
        with open(out_file_path, "wb") as f:
            # looping over the directories and skipping files or directories in the wrong format
            # directories need to end with a digit

            for path in sorted_chunk_dirs:
                src_path = path / filename
                logger.debug(f"Source file: {src_path}")
                if src_path.exists():
                    shutil.copyfileobj(open(src_path, "rb"), f)
                else:
                    # Differing files between the 'chunk' directories are skipped
                    logger.warning(f"The file '{src_path}' does not exist. Not joining")


@click.command(context_settings={"show_default": True})
@click.option(
    "--input_dir",
    required=True,
    help="Folder containing different subfolders with the data chunks.",
)
@click.option("--output_dir", required=True, help="Where to save all the files.")
def main(input_dir: str, output_dir: str) -> None:
    """
    Joins files which were before splitted with the script ensure_data_dimension.py
    """
    setup_console_logger()

    join_data_files(input_dir, output_dir)


if __name__ == "__main__":
    main()
