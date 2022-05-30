import tempfile
from pathlib import Path

import pytest
from rxn_chemutils.tokenization import TokenizationError
from rxn_utilities.file_utilities import dump_list_to_file, load_list_from_file

from rxn_onmt_utils.rxn_models.tokenize_file import (
    classification_file_is_tokenized,
    classification_string_is_tokenized,
    detokenize_class,
    file_is_tokenized,
    tokenize_class,
    tokenize_classification_file,
)


def test_file_is_tokenized():
    # to test file_is_tokenized(), we create a few temporary files in a directory
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_path = Path(temporary_dir)

        # Basic tokenized example
        dump_list_to_file(["C C O >> C C O", "C C . C"], temporary_path / "a.txt")
        assert file_is_tokenized(temporary_path / "a.txt")

        # Basic non-tokenized example
        dump_list_to_file(["CCO>>CCO", "CC.C"], temporary_path / "b.txt")
        assert not file_is_tokenized(temporary_path / "b.txt")

        # Only checks the first line - returns True even if the second line is not tokenized
        dump_list_to_file(["C C O >> C C O", "CC.C"], temporary_path / "c.txt")
        assert file_is_tokenized(temporary_path / "c.txt")

        # empty file
        dump_list_to_file([], temporary_path / "d.txt")
        with pytest.raises(StopIteration):
            _ = file_is_tokenized(temporary_path / "d.txt")

        # Invalid SMILES
        dump_list_to_file(["I N V A L I D", "CC.C"], temporary_path / "e.txt")
        with pytest.raises(TokenizationError):
            _ = file_is_tokenized(temporary_path / "e.txt")


def test_tokenize_class():
    assert tokenize_class("1.2.3") == "1 1.2 1.2.3"
    assert tokenize_class("0") == "0"

    # if already tokenized, return the input
    assert tokenize_class("1 1.2 1.2.3") == "1 1.2 1.2.3"

    # Tokenization errors
    with pytest.raises(ValueError):
        _ = tokenize_class("1.2.3.4")
        _ = tokenize_class("1~2~3")


def test_detokenize_class():
    assert detokenize_class("1 1.2 1.2.3") == "1.2.3"
    assert detokenize_class("0") == "0"

    # if already detokenized, return the input
    assert detokenize_class("1.2.3") == "1.2.3"

    # Tokenization errors
    with pytest.raises(ValueError):
        _ = detokenize_class("1.2.3.4")
        _ = detokenize_class("1~2~3")
        _ = detokenize_class("1 1.2")
        _ = detokenize_class("1 1.2 1.2.3 1.2.3.4")


def test_classification_string_is_tokenized():
    assert classification_string_is_tokenized("1 1.2 1.2.3")
    assert not classification_string_is_tokenized("1.2.3")

    # Special case - for unrecognized reactions
    assert classification_string_is_tokenized("0")

    # Tokenization errors are being propagated
    with pytest.raises(ValueError):
        _ = classification_string_is_tokenized("1 1.2")


def test_classification_file_is_tokenized():
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_path = Path(temporary_dir)

        # Basic tokenized example
        dump_list_to_file(["1 1.2 1.2.3", "0"], temporary_path / "a.txt")
        assert classification_file_is_tokenized(temporary_path / "a.txt")

        # Basic non-tokenized example
        dump_list_to_file(["1.2.3", "0"], temporary_path / "b.txt")
        assert not classification_file_is_tokenized(temporary_path / "b.txt")

        # Only checks the first line - returns True even if the second line is not tokenized
        dump_list_to_file(["1 1.2 1.2.3", "1.2.3"], temporary_path / "c.txt")
        assert classification_file_is_tokenized(temporary_path / "c.txt")

        # empty file
        dump_list_to_file([], temporary_path / "d.txt")
        with pytest.raises(StopIteration):
            _ = classification_file_is_tokenized(temporary_path / "d.txt")

        # Invalid
        dump_list_to_file(["1 1.2", "1 1.2 1.2.3"], temporary_path / "e.txt")
        with pytest.raises(ValueError):
            _ = classification_file_is_tokenized(temporary_path / "e.txt")


def test_tokenize_classification_file():
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_path = Path(temporary_dir)

        # Basic tokenized example
        dump_list_to_file(["1.2.3", "0", "2.11.4", "11.0.23"], temporary_path / "a.txt")
        tokenize_classification_file(
            temporary_path / "a.txt", temporary_path / "a.txt.tokenized"
        )
        assert load_list_from_file(temporary_path / "a.txt.tokenized") == [
            "1 1.2 1.2.3",
            "0",
            "2 2.11 2.11.4",
            "11 11.0 11.0.23",
        ]

        # Invalid classes present
        dump_list_to_file(
            ["1.2.3.5", "0", "2.11.4", "11.0.23"], temporary_path / "b.txt"
        )
        tokenize_classification_file(
            temporary_path / "b.txt", temporary_path / "b.txt.tokenized"
        )
        assert load_list_from_file(temporary_path / "b.txt.tokenized") == [
            "",
            "0",
            "2 2.11 2.11.4",
            "11 11.0 11.0.23",
        ]
