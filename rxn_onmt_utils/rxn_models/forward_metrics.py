from typing import Any, Dict, Iterable

from rxn.utilities.files import PathLike, iterate_lines_from_file

from .metrics import top_n_accuracy
from .metrics_calculator import MetricsCalculator
from .metrics_files import ForwardFiles, MetricsFiles


class ForwardMetrics(MetricsCalculator):
    """
    Class to compute common metrics for forward models, starting from files
    containing the ground truth and predictions.

    Note: all files are expected to be standardized (canonicalized, sorted, etc.).
    """

    def __init__(self, gt_products: Iterable[str], predicted_products: Iterable[str]):
        self.gt_products = list(gt_products)
        self.predicted_products = list(predicted_products)

    def get_metrics(self) -> Dict[str, Any]:
        topn = top_n_accuracy(
            ground_truth=self.gt_products, predictions=self.predicted_products
        )

        return {"accuracy": topn}

    @classmethod
    def from_metrics_files(cls, metrics_files: MetricsFiles) -> "ForwardMetrics":
        if not isinstance(metrics_files, ForwardFiles):
            raise ValueError("Invalid type provided")
        return cls.from_raw_files(
            gt_products_file=metrics_files.gt_tgt,
            predicted_products_file=metrics_files.predicted_canonical,
        )

    @classmethod
    def from_raw_files(
        cls,
        gt_products_file: PathLike,
        predicted_products_file: PathLike,
    ) -> "ForwardMetrics":
        return cls(
            gt_products=iterate_lines_from_file(gt_products_file),
            predicted_products=iterate_lines_from_file(predicted_products_file),
        )
