from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from rxn.chemutils.reaction_smiles import parse_any_reaction_smiles
from rxn.utilities.containers import chunker
from rxn.utilities.files import PathLike, iterate_lines_from_file

from .metrics import get_sequence_multiplier, top_n_accuracy
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
        partial_match = fraction_of_identical_compounds(
            ground_truth=self.gt_tgt, predictions=self.predicted_context
        )

        return {"accuracy": topn, "partial_match": partial_match}

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


def identical_fraction(ground_truth: str, prediction: str) -> float:
    """For context prediction models, fraction of compounds that are identical to
    the ground truth.

    The concept of overlap is hard to define uniquely; this is a tentative
    implementation for getting an idea of how the models behave.

    As denominator, takes the size of whichever list is larger."""
    try:
        gt_reaction = parse_any_reaction_smiles(ground_truth)
        pred_reaction = parse_any_reaction_smiles(prediction)

        n_compounds_tot = 0
        n_compounds_match = 0

        for gt_group, pred_group in zip(gt_reaction, pred_reaction):
            gt_compounds = set(gt_group)
            pred_compounds = set(pred_group)
            overlap = gt_compounds.intersection(pred_compounds)
            n_compounds_tot += max(len(gt_compounds), len(pred_compounds))
            n_compounds_match += len(overlap)

        if n_compounds_tot == 0:
            return 1.0
        return n_compounds_match / n_compounds_tot

    except Exception:
        return 0.0


def fraction_of_identical_compounds(
    ground_truth: Sequence[str], predictions: Sequence[str]
) -> Dict[int, float]:
    """
    Compute the fraction of identical compounds, split by n-th predictions.

    Raises:
        ValueError: if the list sizes are incompatible, forwarded from get_sequence_multiplier().

    Returns:
        Dictionary for the fraction of identical compounds, by top-n.
    """
    multiplier = get_sequence_multiplier(
        ground_truth=ground_truth, predictions=predictions
    )

    # we will get, for each prediction of each "n", the portion that is matching
    overlap_for_n: List[List[float]] = [[] for _ in range(multiplier)]

    # We will process sample by sample - for that, we need to chunk the predictions
    prediction_chunks = chunker(predictions, chunk_size=multiplier)
    for gt, predictions in zip(ground_truth, prediction_chunks):
        for i, prediction in enumerate(predictions):
            overlap = identical_fraction(gt, prediction)
            overlap_for_n[i].append(overlap)

    accuracy = {i + 1: np.mean(overlap_for_n[i]) for i in range(multiplier)}
    return accuracy
