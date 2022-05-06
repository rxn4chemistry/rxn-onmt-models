import tempfile
from pathlib import Path

import pytest
from rxn_chemutils.tokenization import TokenizationError
from rxn_utilities.file_utilities import dump_list_to_file

from rxn_onmt_utils.rxn_models.utils import string_is_tokenized, file_is_tokenized


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
