import re
from pathlib import Path
from typing import Union, Optional


class ModelFiles:
    """
    Class to make it easy to get the names/paths of the trained OpenNMT models.
    """
    ONMT_CONFIG_FILE = 'config.yml'
    MODEL_PREFIX = 'model'
    MODEL_STEP_PATTERN = re.compile(r'model_step_(\d+)\.pt')

    def __init__(self, model_dir: Union[Path, str]):
        # Directly converting to an absolute path
        self.model_dir = Path(model_dir).resolve()
        # Create the directory if it does not exist yet
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_prefix(self) -> Path:
        """Absolute path to the model prefix; during training, OpenNMT will
        append "_step_10000.pt" to it (or other step numbers)."""
        return self.model_dir / ModelFiles.MODEL_PREFIX

    @property
    def config_file(self) -> Path:
        """Absolute path to the model prefix; during training, OpenNMT will
        append "_step_10000.pt" to it (or other step numbers)."""
        return self.model_dir / ModelFiles.ONMT_CONFIG_FILE

    def get_last_checkpoint(self) -> Path:
        """Get the last checkpoint matching the naming including the step number.

        Raises:
            RuntimeError: no model is found in the expected directory.
        """
        models_and_steps = [
            (self._get_checkpoint_step(path), path) for path in self.model_dir.iterdir()
        ]
        models_and_steps = [(step, path) for step, path in models_and_steps if step is not None]
        if not models_and_steps:
            raise RuntimeError(f'No model found in "{self.model_dir}"')

        # Reverse sort, get the path of the first item.
        return sorted(models_and_steps, reverse=True)[0][1]

    def _get_checkpoint_step(self, path: Path) -> Optional[int]:
        """Get the step from the path of a given model. None if no match."""
        match = ModelFiles.MODEL_STEP_PATTERN.match(path.name)
        if match is None:
            return None
        return int(match.group(1))


class OnmtPreprocessedFiles:
    """
    Class to make it easy to get the names/paths of the OpenNMT-preprocessed files.
    """
    PREFIX = 'preprocessed'

    def __init__(self, preprocessed_dir: Union[Path, str]):
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
        return self.preprocess_prefix.with_suffix('.vocab.pt')


class RxnPreprocessingFiles:
    """
    Class to make it easy to get the names/paths of the files generated during data preprocessing.

    This assumes that the default paths were used when calling rxn-data-pipeline.
    """
    FILENAME_ROOT = 'data'

    def __init__(self, processed_data_dir: Union[Path, str]):
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
        if not extension.startswith('.'):
            extension = '.' + extension
        return self.processed_data_dir / (RxnPreprocessingFiles.FILENAME_ROOT + extension)

    @property
    def standardized_csv(self) -> Path:
        return self._add_extension('standardized.csv')

    @property
    def processed_csv(self) -> Path:
        return self._add_extension('processed.csv')

    @property
    def processed_train_csv(self) -> Path:
        return self._add_extension('processed.train.csv')

    @property
    def processed_validation_csv(self) -> Path:
        return self._add_extension('processed.validation.csv')

    @property
    def processed_test_csv(self) -> Path:
        return self._add_extension('processed.test.csv')

    @property
    def train_precursors(self) -> Path:
        return self._add_extension('processed.train.precursors_tokens')

    @property
    def train_products(self) -> Path:
        return self._add_extension('processed.train.products_tokens')

    @property
    def validation_precursors(self) -> Path:
        return self._add_extension('processed.validation.precursors_tokens')

    @property
    def validation_products(self) -> Path:
        return self._add_extension('processed.validation.products_tokens')

    @property
    def test_precursors(self) -> Path:
        return self._add_extension('processed.test.precursors_tokens')

    @property
    def test_products(self) -> Path:
        return self._add_extension('processed.test.products_tokens')

    def get_tokenized_src_file(self, split: str, model_task: str) -> Path:
        if split == 'train' and model_task == 'forward':
            return self.train_precursors
        if split == 'train' and model_task == 'retro':
            return self.train_products
        if split == 'valid' and model_task == 'forward':
            return self.validation_precursors
        if split == 'valid' and model_task == 'retro':
            return self.validation_products
        if split == 'test' and model_task == 'forward':
            return self.test_precursors
        if split == 'test' and model_task == 'retro':
            return self.test_products
        raise ValueError(f'Unsupported combination: "{split}", "{model_task}"')

    def get_tokenized_tgt_file(self, split: str, model_task: str) -> Path:
        if split == 'train' and model_task == 'forward':
            return self.train_products
        if split == 'train' and model_task == 'retro':
            return self.train_precursors
        if split == 'valid' and model_task == 'forward':
            return self.validation_products
        if split == 'valid' and model_task == 'retro':
            return self.validation_precursors
        if split == 'test' and model_task == 'forward':
            return self.test_products
        if split == 'test' and model_task == 'retro':
            return self.test_precursors
        raise ValueError(f'Unsupported combination: "{split}", "{model_task}"')
