import logging
import re
from itertools import count
from pathlib import Path
from typing import Optional

from rxn.utilities.files import PathLike

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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

    @staticmethod
    def augmented(data_path: Path) -> Path:
        """Get the path for the augmented version of a data file."""
        return data_path.with_name(data_path.name + ".augmented")

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
