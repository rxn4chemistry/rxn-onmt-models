from typing import List, Dict, Sequence, TypeVar

from rxn_utilities.container_utilities import chunker

T = TypeVar('T')


def top_n_accuracy(ground_truth: Sequence[T], predictions: Sequence[T]) -> Dict[int, float]:
    """
    Compute the top-n accuracy values.

    Raises:
        ValueError: if the list sizes are incompatible, forwarded from get_multiplier().

    Returns:
        Dictionary of top-n accuracy values.
    """
    multiplier = get_multiplier(ground_truth=ground_truth, predictions=predictions)

    # we will count, for each "n", how many predictions are correct
    correct_for_topn: List[int] = [0 for _ in range(multiplier)]

    # We will process sample by sample - for that, we need to chunk the predictions
    prediction_chunks = chunker(predictions, chunk_size=multiplier)

    for gt, predictions in zip(ground_truth, prediction_chunks):
        for i in range(multiplier):
            correct = gt in predictions[:i + 1]
            correct_for_topn[i] += int(correct)

    return {i + 1: correct_for_topn[i] / len(ground_truth) for i in range(multiplier)}


def round_trip_accuracy(ground_truth: Sequence[T], predictions: Sequence[T]) -> Dict[int, float]:
    """
    Compute the round-trip accuracy values, split by n-th predictions.

    Raises:
        ValueError: if the list sizes are incompatible, forwarded from get_multiplier().

    Returns:
        Dictionary of round-trip "n" accuracy values.
    """
    multiplier = get_multiplier(ground_truth=ground_truth, predictions=predictions)

    # we will count, for each "n", how many predictions are correct
    correct_for_n: List[int] = [0 for _ in range(multiplier)]

    # We will process sample by sample - for that, we need to chunk the predictions
    prediction_chunks = chunker(predictions, chunk_size=multiplier)

    for gt, predictions in zip(ground_truth, prediction_chunks):
        correct_values = 0
        for i, prediction in enumerate(predictions):
            correct = gt == prediction
            correct_values += int(correct)
            correct_for_n[i] += correct_values

    # Note: the total number of predictions to take into account for the "n"-th (= "i+1"th)
    # value is "n * len(ground_truth)".
    return {i + 1: correct_for_n[i] / ((i + 1) * len(ground_truth)) for i in range(multiplier)}


def get_multiplier(ground_truth: Sequence[T], predictions: Sequence[T]) -> int:
    """
    Get the multiplier for the number of predictions by ground truth sample.

    Raises:
        ValueError: if the lists have inadequate sizes
    """
    n_gt = len(ground_truth)
    n_pred = len(predictions)

    if n_gt < 1 or n_pred < 1:
        raise ValueError(
            f'Inadequate number of predictions ({n_pred}) and/or ground truth samples ({n_gt})'
        )

    multiplier = n_pred // n_gt

    if n_pred != multiplier * n_gt:
        raise ValueError(
            f'The number of predictions ({n_pred}) is not an exact '
            f'multiple of the number of ground truth samples ({n_gt})'
        )

    return multiplier
