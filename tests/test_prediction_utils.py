from pathlib import Path
from typing import Iterable, Iterator, Optional

import pytest
from rxn.utilities.files import dump_list_to_file, named_temporary_path

from rxn.onmt_models.prediction_utils import MultiPrediction, load_predictions


class PredFiles:
    """Helper class to hold temporary files for the tests below."""

    def __init__(self, temporary_path: Path):
        temporary_path.mkdir()
        self.src = temporary_path / "src.txt"
        self.pred = temporary_path / "pred.txt"
        self.tgt = temporary_path / "tgt.txt"


def _generate_temporary_files(
    temporary_files: PredFiles,
    src: Iterable[str],
    pred: Iterable[str],
    tgt: Optional[Iterable[str]],
) -> None:
    """Helper function to generate files for the tests below."""
    dump_list_to_file(src, temporary_files.src)
    dump_list_to_file(pred, temporary_files.pred)
    if tgt is not None:
        dump_list_to_file(tgt, temporary_files.tgt)


@pytest.fixture
def temp_files() -> Iterator[PredFiles]:
    """Fixture to use in the tests below to create a directory and return
    it via an instance of PredFiles."""
    with named_temporary_path() as temp_path:
        yield PredFiles(temp_path)


def test_load_predictions_without_tgt(temp_files: PredFiles) -> None:
    _generate_temporary_files(
        temp_files,
        src=["A", "B", "C"],
        pred=["a1", "a2", "b1", "b2", "c1", "c2"],
        tgt=None,
    )
    predictions = load_predictions(temp_files.src, pred_file=temp_files.pred)
    assert list(predictions) == [
        MultiPrediction("A", ["a1", "a2"], None),
        MultiPrediction("B", ["b1", "b2"], None),
        MultiPrediction("C", ["c1", "c2"], None),
    ]


def test_load_predictions_with_tgt(temp_files: PredFiles) -> None:
    _generate_temporary_files(
        temp_files,
        src=["A", "B", "C"],
        pred=["a1", "a2", "a3", "b1", "b2", "b3", "c1", "c2", "c3"],
        tgt=["tA", "tB", "tC"],
    )
    predictions = load_predictions(
        temp_files.src, pred_file=temp_files.pred, tgt_file=temp_files.tgt
    )
    assert list(predictions) == [
        MultiPrediction("A", ["a1", "a2", "a3"], "tA"),
        MultiPrediction("B", ["b1", "b2", "b3"], "tB"),
        MultiPrediction("C", ["c1", "c2", "c3"], "tC"),
    ]


def test_load_predictions_file_length_mismatch(temp_files: PredFiles) -> None:
    # No exception raised if the file lengths don't match at this time.
    # We give an explicit multiplier to disable its automatic detection
    _generate_temporary_files(
        temp_files,
        src=["A", "B", "C", "D", "E", "F", "G"],
        pred=["a", "b", "c"],
        tgt=["tA", "tB", "tC"],
    )
    predictions = load_predictions(
        temp_files.src, pred_file=temp_files.pred, tgt_file=temp_files.tgt, multiplier=1
    )
    # to observe: no D/E/F although present in the source file.
    assert list(predictions) == [
        MultiPrediction("A", ["a"], "tA"),
        MultiPrediction("B", ["b"], "tB"),
        MultiPrediction("C", ["c"], "tC"),
    ]
