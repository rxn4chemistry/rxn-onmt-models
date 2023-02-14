import logging
from collections import defaultdict
from functools import partial
from typing import Callable, DefaultDict, Iterable, Iterator, Tuple

from rxn.chemutils.conversion import canonicalize_smiles, smiles_to_inchi
from rxn.chemutils.miscellaneous import apply_to_any_smiles, sort_any
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# We handle predictions as tuples of SMILES strings and associated confidences
Prediction: TypeAlias = Tuple[str, float]


class BaseCollapser:
    """
    Helper class for collapsing things.

    Having this base collapser can be useful for extending things in the future, if we
    want to make this class more general than just working with tuples.
    """

    def __init__(self, collapsing_fns: Iterable[Callable[[str], str]]):
        self.collapsing_fns = list(collapsing_fns)

    def collapse(self, predictions: Iterable[Prediction]) -> Iterator[Prediction]:
        # Convert to a list, as we will need to iterate two times over it
        predictions = list(predictions)

        collapsed_mapping = dict()
        collapsed_confidence: DefaultDict[str, float] = defaultdict(float)

        for smiles, confidence in predictions:
            collapsed_smiles = sort_any(smiles)
            for fn in self.collapsing_fns:
                try:
                    collapsed_smiles = apply_to_any_smiles(collapsed_smiles, fn)
                except Exception as e:
                    logger.warning(f'Cannot collapse SMILES "{smiles}": {e}')
            collapsed_mapping[smiles] = collapsed_smiles
            collapsed_confidence[collapsed_smiles] += confidence

        consumed = set()

        for smiles, _ in predictions:
            collapsed_smiles = collapsed_mapping[smiles]

            # Check if the collapsed representation has already been seen
            if collapsed_smiles in consumed:
                continue
            consumed.add(collapsed_smiles)

            yield smiles, collapsed_confidence[collapsed_smiles]


class PredictionCollapser:
    """
    Collapse the predictions of an RXN-onmt model based on canonical representations
    of the predictions.

    This is useful to remove predictions that are different in the raw string,
    but correspond to identical compounds.
    """

    def __init__(self, collapse_inchi: bool = True):
        """
        Args:
            collapse_inchi: whether to do the collapsing based on the InChI.
        """
        self.collapser = self._instantiate_base_collapser(collapse_inchi)

    @staticmethod
    def _instantiate_base_collapser(collapse_inchi: bool) -> BaseCollapser:
        canonicalize = partial(canonicalize_smiles, check_valence=False)
        collapsing_fns = [canonicalize]

        if collapse_inchi:
            to_inchi = partial(smiles_to_inchi, extended_tautomer_check=True)
            collapsing_fns.append(to_inchi)

        return BaseCollapser(collapsing_fns=collapsing_fns)

    def collapse_predictions(
        self, predictions: Iterable[Prediction]
    ) -> Iterator[Prediction]:
        yield from self.collapser.collapse(predictions)
