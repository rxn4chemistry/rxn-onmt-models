import logging
import os
import re
from enum import Flag
from itertools import count
from pathlib import Path
from typing import List, Optional

from rxn.chemutils.tokenization import detokenize_smiles, to_tokens
from rxn.utilities.files import PathLike

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class UnclearWhetherTokenized(ValueError):
    """Exception raised when unclear if something was tokenized or not."""

    def __init__(self, string: str):
        super().__init__(f'Cannot determine if "{string}" is tokenized.')


def convert_class_token_idx_for_tranlation_models(class_token_idx: int) -> str:
    return f"[{class_token_idx}]"


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
    Whether a string is a tokenized SMILES or not.

    Args:
        smiles_line: string to inspect

    Raises:
        ValueError: if not possible to determine whether tokenized or not
        TokenizationError: propagated directly from tokenize_smiles()
    """
    detokenized = detokenize_smiles(smiles_line)
    tokens = to_tokens(detokenized)
    if len(tokens) < 2:
        raise UnclearWhetherTokenized(smiles_line)
    return " ".join(tokens) == smiles_line


class MetricsFiles:
    def __init__(self, directory: PathLike):
        self.directory = Path(directory)
        self.log_file = self.directory / "log.txt"
        self.metrics_file = self.directory / "metrics.json"


class RetroFiles(MetricsFiles):
    """
    Class holding the locations of the files to write to or to read from for
    the evaluation of retro metrics.
    """

    REORDERED_FILE_EXTENSION = ".reordered"

    def __init__(self, directory: PathLike):
        super().__init__(directory=directory)
        self.gt_products = self.directory / "gt_products.txt"
        self.gt_precursors = self.directory / "gt_precursors.txt"
        self.class_token_products = self.directory / "class_token_products.txt"
        self.class_token_precursors = self.directory / "class_token_precursors.txt"
        self.predicted_precursors = self.directory / "predicted_precursors.txt"
        self.predicted_precursors_canonical = (
            self.directory / "predicted_precursors_canonical.txt"
        )
        self.predicted_precursors_log_probs = (
            self.directory / "predicted_precursors.txt.tokenized_log_probs"
        )
        self.predicted_products = self.directory / "predicted_products.txt"
        self.predicted_products_canonical = (
            self.directory / "predicted_products_canonical.txt"
        )
        self.predicted_products_log_probs = (
            self.directory / "predicted_products.txt.tokenized_log_probs"
        )
        self.predicted_rxn_canonical = self.directory / "predicted_rxn_canonical.txt"
        self.predicted_classes = self.directory / "predicted_classes.txt"


class ForwardFiles(MetricsFiles):
    """
    Class holding the locations of the files to write to or to read from for
    the evaluation of forward metrics.
    """

    def __init__(self, directory: PathLike):
        super().__init__(directory=directory)
        self.gt_products = self.directory / "gt_products.txt"
        self.gt_precursors = self.directory / "gt_precursors.txt"
        self.predicted_products = self.directory / "predicted_products.txt"
        self.predicted_products_canonical = (
            self.directory / "predicted_products_canonical.txt"
        )


def preprocessed_id_names(n_additional_sets: int) -> List[str]:
    """Get the names of the ids for the datasets used in multi-task training
    with OpenNMT.

    Args:
        n_additional_sets: how many sets there are in addition to the main set.
    """
    return ["main_set"] + [f"additional_set_{i+1}" for i in range(n_additional_sets)]


class ModelFiles:
    """
    Class to make it easy to get the names/paths of the trained OpenNMT models.
    """

    ONMT_CONFIG_FILE = "config_{idx}.yml"
    MODEL_PREFIX = "model"
    MODEL_STEP_PATTERN = re.compile(r"^model_step_(\d+)\.pt$")

    def __init__(self, model_dir: PathLike):
        # Directly converting to an absolute path
        self.model_dir = Path(model_dir).resolve()
        # Create the directory if it does not exist yet
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_prefix(self) -> Path:
        """Absolute path to the model prefix; during training, OpenNMT will
        append "_step_10000.pt" to it (or other step numbers)."""
        return self.model_dir / ModelFiles.MODEL_PREFIX

    def next_config_file(self) -> Path:
        """Get the next available config file name."""
        for idx in count(1):
            config_file = self.model_dir / ModelFiles.ONMT_CONFIG_FILE.format(idx=idx)
            if not config_file.exists():
                return config_file
        return Path()  # Note: in order to satisfy mypy. This is never reached.

    def get_last_checkpoint(self) -> Path:
        """Get the last checkpoint matching the naming including the step number.

        Raises:
            RuntimeError: no model is found in the expected directory.
        """
        models_and_steps = [
            (self._get_checkpoint_step(path), path) for path in self.model_dir.iterdir()
        ]
        models_and_steps = [
            (step, path) for step, path in models_and_steps if step is not None
        ]
        if not models_and_steps:
            raise RuntimeError(f'No model found in "{self.model_dir}"')

        # Reverse sort, get the path of the first item.
        return sorted(models_and_steps, reverse=True)[0][1]

    @staticmethod
    def _get_checkpoint_step(path: Path) -> Optional[int]:
        """Get the step from the path of a given model. None if no match."""
        match = ModelFiles.MODEL_STEP_PATTERN.match(path.name)
        if match is None:
            return None
        return int(match.group(1))


class OnmtPreprocessedFiles:
    """
    Class to make it easy to get the names/paths of the OpenNMT-preprocessed files.
    """

    PREFIX = "preprocessed"

    def __init__(self, preprocessed_dir: PathLike):
        # Directly converting to an absolute path
        self.preprocessed_dir = Path(preprocessed_dir).resolve()
        # Create the directory if it does not exist yet
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def preprocess_prefix(self) -> Path:
        """Absolute path to the prefix for the preprocessed files; during preprocessing,
        OpenNMT will append ".train.0.pt", ".valid.0.pt", ".vocab.pt", etc."""
        return self.preprocessed_dir / OnmtPreprocessedFiles.PREFIX

    @property
    def vocab_file(self) -> Path:
        return self.preprocess_prefix.with_suffix(".vocab.pt")


class RxnPreprocessingFiles:
    """
    Class to make it easy to get the names/paths of the files generated during data preprocessing.

    This assumes that the default paths were used when calling rxn-data-pipeline.
    """

    FILENAME_ROOT = "data"

    def __init__(self, processed_data_dir: PathLike):
        # Directly converting to an absolute path
        self.processed_data_dir = Path(processed_data_dir).resolve()

    def _add_extension(self, extension: str) -> Path:
        """
        Helper function get the path of the file produced with the given extension.

        Args:
            extension: extension to add

        Returns:
            Path to the file with the given extension.
        """
        if not extension.startswith("."):
            extension = "." + extension
        return self.processed_data_dir / (
            RxnPreprocessingFiles.FILENAME_ROOT + extension
        )

    @property
    def standardized_csv(self) -> Path:
        return self._add_extension("standardized.csv")

    @property
    def processed_csv(self) -> Path:
        return self._add_extension("processed.csv")

    def get_processed_csv_for_split(self, split: str) -> Path:
        split = self._validate_split(split)
        return self._add_extension(f"processed.{split}.csv")

    @property
    def processed_train_csv(self) -> Path:
        return self.get_processed_csv_for_split("train")

    @property
    def processed_validation_csv(self) -> Path:
        return self.get_processed_csv_for_split("validation")

    @property
    def processed_test_csv(self) -> Path:
        return self.get_processed_csv_for_split("test")

    def get_precursors_for_split(self, split: str) -> Path:
        split = self._validate_split(split)
        return self._add_extension(f"processed.{split}.precursors_tokens")

    def get_products_for_split(self, split: str) -> Path:
        split = self._validate_split(split)
        return self._add_extension(f"processed.{split}.products_tokens")

    @property
    def train_precursors(self) -> Path:
        return self.get_precursors_for_split("train")

    @property
    def train_products(self) -> Path:
        return self.get_products_for_split("train")

    @property
    def validation_precursors(self) -> Path:
        return self.get_precursors_for_split("validation")

    @property
    def validation_products(self) -> Path:
        return self.get_products_for_split("validation")

    @property
    def test_precursors(self) -> Path:
        return self.get_precursors_for_split("test")

    @property
    def test_products(self) -> Path:
        return self.get_products_for_split("test")

    def get_context_tags_for_split(self, split: str) -> Path:
        split = self._validate_split(split)
        return self._add_extension(f"processed.{split}.context.tagged")

    def get_context_src_for_split(self, split: str) -> Path:
        split = self._validate_split(split)
        return self._add_extension(f"processed.{split}.context.src")

    def get_context_tgt_for_split(self, split: str) -> Path:
        split = self._validate_split(split)
        return self._add_extension(f"processed.{split}.context.tgt")

    def _validate_split(self, split: str) -> str:
        if split == "train":
            return "train"
        if split == "valid" or split == "validation":
            return "validation"
        if split == "test":
            return "test"
        raise ValueError(f'Unsupported split: "{split}"')

    def get_src_file(self, split: str, model_task: str) -> Path:
        """Get the source file for the given task.

        Note: the file is tokenized for the forward and retro tasks, but not
        for the context task.
        """
        if model_task == "forward":
            return self.get_precursors_for_split(split)
        if model_task == "retro":
            return self.get_products_for_split(split)
        if model_task == "context":
            return self.get_context_src_for_split(split)
        raise ValueError(f'Unsupported model task: "{model_task}"')

    def get_tgt_file(self, split: str, model_task: str) -> Path:
        """Get the target file for the given task.

        Note: the file is tokenized for the forward and retro tasks, but not
        for the context task.
        """
        if model_task == "forward":
            return self.get_products_for_split(split)
        if model_task == "retro":
            return self.get_precursors_for_split(split)
        if model_task == "context":
            return self.get_context_tgt_for_split(split)
        raise ValueError(f'Unsupported model task: "{model_task}"')


class RxnCommand(Flag):
    """
    Flag indicating which command(s) the parameters relate to.

    TC, TF, TCF are the combinations of the three base flags.
    This enum allows for easily checking which commands some parameters relate
    to (see Parameter and TrainingPlanner classes).
    """

    T = 1  # Train
    C = 2  # Continue training
    F = 4  # Fine-tune
    TC = 3
    TF = 5
    CF = 6
    TCF = 7
