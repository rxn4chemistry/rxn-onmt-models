import logging
from pathlib import Path

import click
from rxn.onmt_utils import __version__ as onmt_utils_version
from rxn.reaction_preprocessing.config import (
    CommonConfig,
    Config,
    DataConfig,
    FragmentBond,
    InitialDataFormat,
    RxnImportConfig,
    SplitConfig,
    StandardizeConfig,
)
from rxn.reaction_preprocessing.main import preprocess_data
from rxn.utilities.logging import setup_console_and_file_logger

from rxn.onmt_models import __version__ as onmt_models_version
from rxn.onmt_models import defaults
from rxn.onmt_models.training_files import RxnPreprocessingFiles
from rxn.onmt_models.utils import log_file_name_from_time

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command(context_settings=dict(show_default=True))
@click.option("--input_data", type=str, required=True, help="Input data TXT or CSV")
@click.option(
    "--import_from",
    type=str,
    default="txt",
    help=(
        'Column to import reaction SMILES from in a CSV. The default, "txt", '
        "means the input file is a simple TXT file."
    ),
)
@click.option(
    "--output_dir",
    type=str,
    required=True,
    help="Directory where to save the generated files",
)
@click.option(
    "--split_seed", default=defaults.SEED, help="Random seed for splitting step"
)
@click.option(
    "--fragment_bond",
    type=click.Choice(["DOT", "TILDE"], case_sensitive=False),
    default="DOT",
)
def main(
    input_data: str,
    import_from: str,
    output_dir: str,
    split_seed: int,
    fragment_bond: str,
) -> None:
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

    # Set up the logs
    log_file = output_dir_path / log_file_name_from_time("rxn-prepare-data")
    setup_console_and_file_logger(log_file)

    logger.info("Prepare reaction data for training with rxn-onmt-models.")
    logger.info(f"rxn-onmt-utils version: {onmt_utils_version}. ")
    logger.info(f"rxn-onmt-models version: {onmt_models_version}. ")

    if import_from == "txt":
        import_config = RxnImportConfig(data_format=InitialDataFormat.TXT)
    else:
        import_config = RxnImportConfig(
            data_format=InitialDataFormat.CSV, input_csv_column_name=import_from
        )

    cfg = Config(
        data=DataConfig(
            path=str(input_data_path),
            proc_dir=str(output_dir_path),
            name=RxnPreprocessingFiles.FILENAME_ROOT,
        ),
        common=CommonConfig(fragment_bond=FragmentBond[fragment_bond]),
        rxn_import=import_config,
        standardize=StandardizeConfig(
            annotation_file_paths=[], discard_unannotated_metals=False
        ),
        split=SplitConfig(hash_seed=split_seed),
    )

    try:
        logger.info("Running the data preprocessing")
        preprocess_data(cfg)
    except Exception as e:
        logger.exception("Error during data preprocessing:")
        raise SystemExit("Error during data preprocessing") from e


if __name__ == "__main__":
    main()
