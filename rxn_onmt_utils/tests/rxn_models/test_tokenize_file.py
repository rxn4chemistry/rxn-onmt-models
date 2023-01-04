import pytest
from rxn.chemutils.tokenization import TokenizationError
from rxn.utilities.files import (
    dump_list_to_file,
    load_list_from_file,
    named_temporary_path,
)

from rxn_onmt_utils.rxn_models.tokenize_file import (
    classification_file_is_tokenized,
    classification_string_is_tokenized,
    detokenize_class,
    detokenize_file,
    ensure_tokenized_file,
    file_is_tokenized,
    tokenize_class,
    tokenize_classification_file,
    tokenize_file,
)


def test_file_is_tokenized():
    # Basic tokenized example
    with named_temporary_path() as path:
        dump_list_to_file(["C C O >> C C O", "C C . C"], path)
        assert file_is_tokenized(path)

    # Basic non-tokenized example
    with named_temporary_path() as path:
        dump_list_to_file(["CCO>>CCO", "CC.C"], path)
        assert not file_is_tokenized(path)

    # Only checks the first line - returns True even if the second line is not tokenized
    with named_temporary_path() as path:
        dump_list_to_file(["C C O >> C C O", "CC.C"], path)
        assert file_is_tokenized(path)

    # empty file
    with named_temporary_path() as path:
        dump_list_to_file([], path)
        with pytest.raises(RuntimeError):
            _ = file_is_tokenized(path)

    # Invalid SMILES
    with named_temporary_path() as path:
        dump_list_to_file(["I N V A L I D", "CC.C"], path)
        with pytest.raises(TokenizationError):
            _ = file_is_tokenized(path)

    # Empty first line - needs to check the second line!
    with named_temporary_path() as path:
        dump_list_to_file(["", "C C O >> C C O"], path)
        assert file_is_tokenized(path)
    with named_temporary_path() as path:
        dump_list_to_file(["", "CCO>>CCO"], path)
        assert not file_is_tokenized(path)

    # First line has one single token - needs to check the second line!
    with named_temporary_path() as path:
        dump_list_to_file([">>", "C C O >> C C O"], path)
        assert file_is_tokenized(path)
    with named_temporary_path() as path:
        dump_list_to_file([">>", "CCO>>CCO"], path)
        assert not file_is_tokenized(path)


def test_tokenize_file():
    with named_temporary_path() as f_in, named_temporary_path() as f_out:
        # Original content
        original = ["CCO>>CCO", "CC.C", "INVALID", "C(NCC)[S]OC"]
        dump_list_to_file(original, f_in)

        # Expected (tokenized) content
        placeholder = "ERROR"
        tokenized = ["C C O >> C C O", "C C . C", placeholder, "C ( N C C ) [S] O C"]

        tokenize_file(f_in, f_out, invalid_placeholder=placeholder)

        assert load_list_from_file(f_out) == tokenized


def test_detokenize_file():
    with named_temporary_path() as f_in, named_temporary_path() as f_out:
        # Original (tokenized) content
        original = ["C C O >> C C O", "C C . C", "C ( N C C ) [S] O C"]
        dump_list_to_file(original, f_in)

        # Expected (detokenized) content
        detokenized = ["CCO>>CCO", "CC.C", "C(NCC)[S]OC"]

        detokenize_file(f_in, f_out)

        assert load_list_from_file(f_out) == detokenized


def test_ensure_tokenized_file():
    with named_temporary_path() as temporary_path:
        temporary_path.mkdir()

        # prepare filenames
        postfix = ".tknz"
        already_tokenized_file = str(temporary_path / "a.txt")
        not_tokenized_file = str(temporary_path / "b.txt")
        updated_tokenized_file = str(temporary_path / "b.txt") + postfix

        # contents (original and expected)
        placeholder = "error"
        tokenized = ["C C O >> C C O", "C C . C", "C ( N C C ) [S] O"]
        not_tokenized = ["CCO>>CCO", "CC.C", "INVALID", "C(N)[S]O"]
        after_tokenization = ["C C O >> C C O", "C C . C", placeholder, "C ( N ) [S] O"]

        # Put into files
        dump_list_to_file(tokenized, already_tokenized_file)
        dump_list_to_file(not_tokenized, not_tokenized_file)

        # ensure for already tokenized - the original unchanged file can be used
        result = ensure_tokenized_file(
            already_tokenized_file, postfix=postfix, invalid_placeholder=placeholder
        )
        assert result == already_tokenized_file
        assert load_list_from_file(result) == tokenized

        # ensure for non-tokenized - a new file was created with tokenized strings
        result = ensure_tokenized_file(
            not_tokenized_file, postfix=postfix, invalid_placeholder=placeholder
        )
        assert result == updated_tokenized_file
        assert load_list_from_file(updated_tokenized_file) == after_tokenization


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
