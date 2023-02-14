import pytest

from rxn.onmt_training.rxn_models.context_metrics import (
    fraction_of_identical_compounds,
    identical_fraction,
)


def test_identical_fraction():
    # Full match
    assert identical_fraction("C.O>>", "C.O>>") == 1.0
    assert identical_fraction("C.O>>N", "C.O>>N") == 1.0

    # Partial match
    assert identical_fraction("C.O.N>>", "C.O>>") == 2 / 3
    assert identical_fraction("C.O>>N", "C.O>>") == 2 / 3
    assert identical_fraction("C.O>>N", "C>>N") == 2 / 3

    # Don't mix up products and precursors
    assert identical_fraction("C.O>>", ">>C.O") == 0.0

    # Empty on one side
    assert identical_fraction(">>", "C>>") == 0.0
    assert identical_fraction("C>>", ">>") == 0.0

    # Empty on both sides
    assert identical_fraction(">>", ">>") == 1.0


def test_identical_fraction_invalid_input():
    # Not a reaction SMILES
    assert identical_fraction("C.O", "C.O") == 0.0
    assert identical_fraction("C.O>>", "") == 0.0


def test_fraction_of_identical_compounds():
    # a few examples for top-1
    assert fraction_of_identical_compounds(
        ["A.B>>", "B.C>>", "E"],
        ["A>>", "B.C>>", "D"],
    ) == {
        1: 0.5
    }  # 0.5, 1, 0
    assert fraction_of_identical_compounds(
        ["A.B>>", "B.C>>", "E>>"],
        ["A.B>>", "B.C>>", "E>>"],
    ) == {1: 1.0}

    # a few examples for top-2
    assert fraction_of_identical_compounds(
        ["A.B>>", "C.D>>", "E.F>>"],
        ["A.B>>", "A>>", "C.D>>", "C>>", "E.F>>", "E>>"],
    ) == {
        1: 1.0,
        2: 0.5,
    }  # first predictions are all 1.0, second ones are half correct

    # raises if not an exact multiple
    with pytest.raises(ValueError):
        _ = fraction_of_identical_compounds(["A>>", "B>>"], ["A>>", "B>>", "C>>"])
