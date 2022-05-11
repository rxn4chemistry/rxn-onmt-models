import logging

from rxn_utilities.file_utilities import PathLike

LOGGER_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def setup_console_logger() -> None:
    logging.basicConfig(format=LOGGER_FORMAT, level='INFO')


def setup_console_and_file_logger(filename: PathLike) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=LOGGER_FORMAT,
        handlers=[logging.FileHandler(filename),
                  logging.StreamHandler()]
    )
