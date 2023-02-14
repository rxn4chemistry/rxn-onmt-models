from pathlib import Path

from rxn.onmt_training.rxn_models.utils import ModelFiles


def test_get_model_checkpoint_step():
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
