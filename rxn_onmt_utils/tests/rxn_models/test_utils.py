from pathlib import Path

import pytest
from rxn.chemutils.tokenization import TokenizationError

from rxn_onmt_utils.rxn_models.utils import ModelFiles, string_is_tokenized


def test_string_is_tokenized():
    assert string_is_tokenized("C C O . [Na+]")
    assert string_is_tokenized("C C O . [Na+] >> C ( O ) Cl")
    assert not string_is_tokenized("C C O . [Na+] >> C (O) Cl")
    assert not string_is_tokenized("CCO")

    # Special case - unclear what the ideal behavior would be
    assert string_is_tokenized("C")

    # Tokenization errors are being propagated
    with pytest.raises(TokenizationError):
        _ = string_is_tokenized("I N V A L I D")


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
