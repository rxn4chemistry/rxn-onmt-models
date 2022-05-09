#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
import subprocess
from pathlib import Path

import click

from rxn_onmt_utils.from_tunerxn.utils import RxnPreprocessingFiles


@click.command(context_settings=dict(show_default=True))
@click.option('--input_data', type=str, required=True, help='Input data TXT')
@click.option(
    '--output_dir', type=str, required=True, help='Directory where to save the generated files'
)
@click.option('--split_seed', default=42, help='Random seed for splitting step')
def prepare_data(input_data: str, output_dir: str, split_seed: int) -> None:
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

    # Running the command below fails if the paths are relative -> make them absolute
    input_data_path = Path(input_data).resolve()
    output_dir_path = Path(output_dir).resolve()

    # make sure that the required output directory exists
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # NB: if the format is CSV, use the following below:
    #   rxn_import.data_format=CSV
    #   rxn_import.input_csv_column_name=rxn_smiles_xxx

    cmd = [
        'rxn-data-pipeline',
        f'data.path={input_data_path}',
        f'data.proc_dir={output_dir_path}',
        f'data.name={RxnPreprocessingFiles.FILENAME_ROOT}',
        'rxn_import.data_format=TXT',
        'standardize.annotation_file_paths=[]',
        'standardize.discard_unannotated_metals=False',
        f'split.hash_seed={split_seed}',
    ]

    cmd_string = ' '.join(cmd)
    print('Running the command:', cmd_string)
    try:
        output = subprocess.check_output(cmd)
        print(output.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print('Error', e.returncode)
        print(e.output.decode('utf-8'))
        raise SystemExit(f'"{cmd_string}" failed.')


if __name__ == "__main__":
    prepare_data()
