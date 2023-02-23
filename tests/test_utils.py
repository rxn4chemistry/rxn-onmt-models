from pathlib import Path

from freezegun import freeze_time

from rxn.onmt_models.training_files import ModelFiles
from rxn.onmt_models.utils import log_file_name_from_time


def test_get_model_checkpoint_step() -> None:
    # Example not fitting the schema
    assert ModelFiles._get_checkpoint_step(Path("dummy")) is None
    assert ModelFiles._get_checkpoint_step(Path("/some/path/dummy")) is None
    assert ModelFiles._get_checkpoint_step(Path("/some/path/dummy.pt")) is None

    # Correct examples
    assert ModelFiles._get_checkpoint_step(Path("model_step_10.pt")) == 10
    assert ModelFiles._get_checkpoint_step(Path("/path/to/model_step_10.pt")) == 10
    assert ModelFiles._get_checkpoint_step(Path("/path/to/model_step_990.pt")) == 990

    # Small mistakes in the name
    assert ModelFiles._get_checkpoint_step(Path("/path/to/model_step990.pt")) is None
    assert ModelFiles._get_checkpoint_step(Path("model_step_10.gt")) is None
    assert ModelFiles._get_checkpoint_step(Path("model_step_10.pt.bak")) is None
    assert ModelFiles._get_checkpoint_step(Path("/path/to/mdl_step_10.pt")) is None


@freeze_time("2018-03-23 15:54:22")
def test_log_file_name_from_time() -> None:
    assert log_file_name_from_time() == "20180323-1554.log"
    assert log_file_name_from_time("dummy_prefix") == "dummy_prefix-20180323-1554.log"
