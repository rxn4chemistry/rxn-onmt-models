from typing import Any, Dict, Iterable

from rxn.utilities.files import PathLike, iterate_lines_from_file

from .metrics import top_n_accuracy
from .metrics_calculator import MetricsCalculator
from .metrics_files import ContextFiles, MetricsFiles


class ContextMetrics(MetricsCalculator):
    """
    Class to compute common metrics for context prediction models, starting from
    files containing the ground truth and predictions.

    Note: all files are expected to be standardized (canonicalized, sorted, etc.).
    """

    def __init__(self, gt_tgt: Iterable[str], predicted_context: Iterable[str]):
        self.gt_tgt = list(gt_tgt)
        self.predicted_context = list(predicted_context)

    def get_metrics(self) -> Dict[str, Any]:
        topn = top_n_accuracy(
            ground_truth=self.gt_tgt, predictions=self.predicted_context
        )

        return {"accuracy": topn}

    @classmethod
    def from_metrics_files(cls, metrics_files: MetricsFiles) -> "ContextMetrics":
        if not isinstance(metrics_files, ContextFiles):
            raise ValueError("Invalid type provided")
        return cls.from_raw_files(
            gt_tgt_file=metrics_files.gt_tgt,
            predicted_context_file=metrics_files.predicted_canonical,
        )

    @classmethod
    def from_raw_files(
        cls,
        gt_tgt_file: PathLike,
        predicted_context_file: PathLike,
    ) -> "ContextMetrics":
        return cls(
            gt_tgt=iterate_lines_from_file(gt_tgt_file),
            predicted_context=iterate_lines_from_file(predicted_context_file),
        )
