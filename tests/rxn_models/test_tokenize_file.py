import pytest
from rxn.utilities.files import (
    dump_list_to_file,
    load_list_from_file,
    named_temporary_path,
)

from rxn.onmt_utils.rxn_models.tokenize_file import (
    classification_file_is_tokenized,
    classification_string_is_tokenized,
    detokenize_class,
    tokenize_class,
    tokenize_classification_file,
)


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
    # Basic tokenized example
    with named_temporary_path() as path:
        dump_list_to_file(["1 1.2 1.2.3", "0"], path)
        assert classification_file_is_tokenized(path)

    # Basic non-tokenized example
    with named_temporary_path() as path:
        dump_list_to_file(["1.2.3", "0"], path)
        assert not classification_file_is_tokenized(path)

    # Only checks the first line - returns True even if the second line is not tokenized
    with named_temporary_path() as path:
        dump_list_to_file(["1 1.2 1.2.3", "1.2.3"], path)
        assert classification_file_is_tokenized(path)

    # empty file
    with named_temporary_path() as path:
        dump_list_to_file([], path)
        with pytest.raises(RuntimeError):
            _ = classification_file_is_tokenized(path)

    # Invalid
    with named_temporary_path() as path:
        dump_list_to_file(["1 1.2", "1 1.2 1.2.3"], path)
        with pytest.raises(ValueError):
            _ = classification_file_is_tokenized(path)


def test_tokenize_classification_file():
    # Basic tokenized example
    with named_temporary_path() as f_in, named_temporary_path() as f_out:
        dump_list_to_file(["1.2.3", "0", "2.11.4", "11.0.23"], f_in)
        tokenize_classification_file(f_in, f_out)
        assert load_list_from_file(f_out) == [
            "1 1.2 1.2.3",
            "0",
            "2 2.11 2.11.4",
            "11 11.0 11.0.23",
        ]

    # Invalid classes present
    with named_temporary_path() as f_in, named_temporary_path() as f_out:
        dump_list_to_file(["1.2.3.5", "0", "2.11.4", "11.0.23"], f_in)
        tokenize_classification_file(f_in, f_out)
        assert load_list_from_file(f_out) == [
            "",
            "0",
            "2 2.11 2.11.4",
            "11 11.0 11.0.23",
        ]
