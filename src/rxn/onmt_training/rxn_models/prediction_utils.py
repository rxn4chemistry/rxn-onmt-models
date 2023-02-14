import itertools
from typing import Iterator, List, Optional

from attr import define
from rxn.utilities.containers import chunker
from rxn.utilities.files import PathLike, count_lines, iterate_lines_from_file
from rxn.utilities.misc import get_multiplier


@define(frozen=True)
class MultiPrediction:
    """
    Holds information on the source and prediction(s) resulting from a translation.

    The object can also hold the target / ground truth (optionally).
    """

    src: str
    pred: List[str]
    tgt: Optional[str]


def load_predictions(
    src_file: PathLike,
    pred_file: PathLike,
    tgt_file: Optional[PathLike] = None,
    multiplier: Optional[int] = None,
) -> Iterator[MultiPrediction]:
    """
    Args:
        src_file: File with the src.
        pred_file: File with the predictions.
        tgt_file: File with the tgt (ground truth).
        multiplier: how many predictions are made per src item. If None, this
            will be determined automatically.

    Returns:
        Iterator over MultiPrediction objects.
    """

    if multiplier is None:
        number_src = count_lines(src_file)
        number_preds = count_lines(pred_file)
        multiplier = get_multiplier(number_src, number_preds)

    srcs = iterate_lines_from_file(src_file)
    preds = iterate_lines_from_file(pred_file)

    tgts: Iterator[Optional[str]] = itertools.repeat(None)
    if tgt_file is not None:
        tgts = iterate_lines_from_file(tgt_file)

    # The prediction file may contain several lines per src line -> chunk it
    pred_chunks = chunker(preds, chunk_size=multiplier)

    for src, pred_list, tgt in zip(srcs, pred_chunks, tgts):
        yield MultiPrediction(src=src, pred=pred_list, tgt=tgt)
