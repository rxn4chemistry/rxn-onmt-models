from typing import Dict, Iterable, Any

from rxn_utilities.file_utilities import iterate_lines_from_file, PathLike

from .metrics import top_n_accuracy, round_trip_accuracy
from .utils import RetroFiles


class RetroMetrics:
    """
    Class to compute common metrics for retro models, starting from files
    containing the ground truth and predictions.

    Note: all files are expected to be standardized (canonicalized, sorted, etc.).
    """

    def __init__(
        self, gt_precursors: Iterable[str], gt_products: Iterable[str],
        predicted_precursors: Iterable[str], predicted_products: Iterable[str]
    ):
        self.gt_products = list(gt_products)
        self.gt_precursors = list(gt_precursors)
        self.predicted_products = list(predicted_products)
        self.predicted_precursors = list(predicted_precursors)

    def get_metrics(self) -> Dict[str, Any]:
        topn = top_n_accuracy(
            ground_truth=self.gt_precursors, predictions=self.predicted_precursors
        )
        roundtrip = round_trip_accuracy(
            ground_truth=self.gt_products, predictions=self.predicted_products
        )

        return {'accuracy': topn, 'round-trip': roundtrip}

    @classmethod
    def from_retro_files(cls, retro_files: RetroFiles) -> 'RetroMetrics':
        return cls.from_raw_files(
            gt_precursors_file=retro_files.gt_precursors,
            gt_products_file=retro_files.gt_products,
            predicted_precursors_file=retro_files.predicted_precursors_canonical,
            predicted_products_file=retro_files.predicted_products_canonical,
        )

    @classmethod
    def from_raw_files(
        cls,
        gt_precursors_file: PathLike,
        gt_products_file: PathLike,
        predicted_precursors_file: PathLike,
        predicted_products_file: PathLike,
    ) -> 'RetroMetrics':
        return cls(
            gt_precursors=iterate_lines_from_file(gt_precursors_file),
            gt_products=iterate_lines_from_file(gt_products_file),
            predicted_precursors=iterate_lines_from_file(predicted_precursors_file),
            predicted_products=iterate_lines_from_file(predicted_products_file),
        )
