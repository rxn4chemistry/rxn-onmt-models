import tempfile
from pathlib import Path

import pytest
from rxn_chemutils.tokenization import TokenizationError
from rxn_utilities.file_utilities import dump_list_to_file

from rxn_onmt_utils.rxn_models.utils import string_is_tokenized, file_is_tokenized, ModelFiles


def test_string_is_tokenized():
    assert string_is_tokenized('C C O . [Na+]')
    assert string_is_tokenized('C C O . [Na+] >> C ( O ) Cl')
    assert not string_is_tokenized('C C O . [Na+] >> C (O) Cl')
    assert not string_is_tokenized('CCO')

    # Special case - unclear what the ideal behavior would be
    assert string_is_tokenized('C')

    # Tokenization errors are being propagated
    with pytest.raises(TokenizationError):
        _ = string_is_tokenized('I N V A L I D')


def test_file_is_tokenized():
    # to test file_is_tokenized(), we create a few temporary files in a directory
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_path = Path(temporary_dir)

        # Basic tokenized example
        dump_list_to_file(['C C O >> C C O', 'C C . C'], temporary_path / 'a.txt')
        assert file_is_tokenized(temporary_path / 'a.txt')

        # Basic non-tokenized example
        dump_list_to_file(['CCO>>CCO', 'CC.C'], temporary_path / 'b.txt')
        assert not file_is_tokenized(temporary_path / 'b.txt')

        # Only checks the first line - returns True even if the second line is not tokenized
        dump_list_to_file(['C C O >> C C O', 'CC.C'], temporary_path / 'c.txt')
        assert file_is_tokenized(temporary_path / 'c.txt')

        # empty file
        dump_list_to_file([], temporary_path / 'd.txt')
        with pytest.raises(StopIteration):
            _ = file_is_tokenized(temporary_path / 'd.txt')

        # Invalid SMILES
        dump_list_to_file(['I N V A L I D', 'CC.C'], temporary_path / 'e.txt')
        with pytest.raises(TokenizationError):
            _ = file_is_tokenized(temporary_path / 'e.txt')


def test_get_model_checkpoint_step():
    # Example not fitting the schema
    assert ModelFiles._get_checkpoint_step(Path('dummy')) is None
    assert ModelFiles._get_checkpoint_step(Path('/some/path/dummy')) is None
    assert ModelFiles._get_checkpoint_step(Path('/some/path/dummy.pt')) is None

    # Correct examples
    assert ModelFiles._get_checkpoint_step(Path('model_step_10.pt')) == 10
    assert ModelFiles._get_checkpoint_step(Path('/path/to/model_step_10.pt')) == 10
    assert ModelFiles._get_checkpoint_step(Path('/path/to/model_step_990.pt')) == 990

    # Small mistakes in the name
    assert ModelFiles._get_checkpoint_step(Path('/path/to/model_step990.pt')) is None
    assert ModelFiles._get_checkpoint_step(Path('model_step_10.gt')) is None
    assert ModelFiles._get_checkpoint_step(Path('model_step_10.pt.bak')) is None
    assert ModelFiles._get_checkpoint_step(Path('/path/to/mdl_step_10.pt')) is None
