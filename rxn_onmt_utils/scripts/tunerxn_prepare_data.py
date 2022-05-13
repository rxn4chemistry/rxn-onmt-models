#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
import logging
from pathlib import Path

import click
from rxn_reaction_preprocessing.config import (
    Config, DataConfig, RxnImportConfig, InitialDataFormat, StandardizeConfig, SplitConfig
)
from rxn_reaction_preprocessing.main import preprocess_data
from rxn_utilities.logging_utilities import setup_console_logger

from rxn_onmt_utils.rxn_models import defaults
from rxn_onmt_utils.from_tunerxn.utils import RxnPreprocessingFiles

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command(context_settings=dict(show_default=True))
@click.option('--input_data', type=str, required=True, help='Input data TXT')
@click.option(
    '--output_dir', type=str, required=True, help='Directory where to save the generated files'
)
@click.option('--split_seed', default=defaults.SEED, help='Random seed for splitting step')
def main(input_data: str, output_dir: str, split_seed: int) -> None:
    """Preprocess the data to generate a dataset for training transformer models.

    The script will automatically generate the following files in output_dir:
        data.imported.csv
        data.standardized.csv
        data.processed.csv
        data.processed.train.csv
        data.processed.validation.csv
        data.processed.test.csv
        data.processed.train.precursors_tokens
        data.processed.train.products_tokens
        data.processed.validation.precursors_tokens
        data.processed.validation.products_tokens
        data.processed.test.precursors_tokens
        data.processed.test.products_tokens
    """
    setup_console_logger()

    # Running the command below fails if the paths are relative -> make them absolute
    input_data_path = Path(input_data).resolve()
    output_dir_path = Path(output_dir).resolve()

    # make sure that the required output directory exists
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # NB: if the format is CSV, use the following below:
    #   rxn_import.data_format=CSV
    #   rxn_import.input_csv_column_name=rxn_smiles_xxx

    cfg = Config(
        data=DataConfig(
            path=str(input_data_path),
            proc_dir=str(output_dir_path),
            name=RxnPreprocessingFiles.FILENAME_ROOT
        ),
        rxn_import=RxnImportConfig(data_format=InitialDataFormat.TXT),
        standardize=StandardizeConfig(annotation_file_paths=[], discard_unannotated_metals=False),
        split=SplitConfig(hash_seed=split_seed),
    )

    try:
        logger.info('Running the data preprocessing')
        preprocess_data(cfg)
    except Exception as e:
        logger.exception('Error during data preprocessing:')
        raise SystemExit('Error during data preprocessing') from e


if __name__ == "__main__":
    main()
