import math
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from rxn.chemutils.conversion import canonicalize_smiles, maybe_canonicalize
from rxn.chemutils.exceptions import InvalidSmiles
from rxn.chemutils.multicomponent_smiles import (
    list_to_multicomponent_smiles,
    multicomponent_smiles_to_list,
)
from rxn.chemutils.tokenization import detokenize_smiles, tokenize_smiles
from rxn.onmt_utils.translator import Translator

from .prediction_collapser import PredictionCollapser

# Aliases, for ease of use. Note that after processing, we consider
# each prediction to contain a list of products.
RawPrediction = Tuple[str, float]
ProcessedPrediction = Tuple[List[str], float]


class ForwardPredictor:
    """
    Helper class to do forward predictions on batches of reactions with
    collapsing of equivalent results.
    """

    def __init__(
        self,
        model: Translator,
        topn: int,
        fragment_bond: str = "~",
        canonicalize_input: bool = True,
        sort_input: bool = True,
        canonicalize_output: bool = True,
        collapse_inchi: bool = True,
    ):
        """
        Args:
            model: forward prediction model.
            topn: how many predictions to make for each input.
            fragment_bond: fragment bond used by the model.
            canonicalize_input: whether to canonicalize the input.
            sort_input: whether to sort the input alphabetically.
            canonicalize_output: whether to canonicalize the output.
            collapse_inchi: whether to collapse predictions having the same InChI.
        """

        self.model = model
        self.fragment_bond = fragment_bond
        self.canonicalize_input = canonicalize_input
        self.sort_input = sort_input

        self.topn = topn
        self.prediction_collapser = PredictionCollapser(collapse_inchi=collapse_inchi)
        self.canonicalize_output = canonicalize_output

    def predict(
        self,
        precursors_lists: Iterable[List[str]],
    ) -> Iterator[Dict[str, Any]]:
        """
        Predict the product for lists of precursors.

        The output contains both the raw predictions and the processed / collapsed
        products.

        This produces, for each list of precursors, a dictionary in the following
        format (here with topn=3, assuming that the two first predictions collapse):

            {
                "results":[
                    {
                        "smiles":[
                            "CCCNC(=O)CC"
                        ],
                        "confidence":0.9999704545228182
                    },
                    {
                        "smiles":[
                            "CCCN.Cl",
                            "O"
                        ],
                        "confidence":1.642442728161003e-07
                    }
                ],
                "raw_results":[
                    {
                        "smiles":"CCCNC(=O)CC",
                        "confidence":0.9999581588476526
                    },
                    {
                        "smiles":"CCCN=C(O)CC",
                        "confidence":1.2295675165578265e-05
                    },
                    {
                        "smiles":"CCCN~Cl.O",
                        "confidence":1.642442728161003e-07
                    }
                ]
            }

        Args:
            precursors_lists: iterable over lists of precursors (one list of precursor
                per reaction to predict). Note: no tildes necessary here as fragment
                bonds, as the precursors are given as lists.

        Returns:
            Iterator over result dictionaries. See format above in this docstring.
        """

        results_iterator = self.predict_batch(precursors_lists=precursors_lists)

        for raw_predictions, processed_predictions in results_iterator:
            yield {
                "results": list(self._wrap_into_dicts(processed_predictions)),
                "raw_results": list(self._wrap_into_dicts(raw_predictions)),
            }

    def predict_batch(
        self,
        precursors_lists: Iterable[List[str]],
    ) -> Iterator[Tuple[List[RawPrediction], List[ProcessedPrediction]]]:
        """
        Predict a batch of reactions with top-n prediction.

        Args:
            precursors_lists: iterable over lists of precursors (one list of precursor
                per reaction to predict). Note: no tildes necessary here as fragment
                bonds, as the precursors are given as lists.

        Returns:
            For each input, a tuple containing 1) the list of raw predictions,
            and 2) the processed/collapsed predictions.
        """
        results_iterator = self.model.translate_multiple_with_scores(
            self.prepare_precursors(precursors_lists), n_best=self.topn
        )

        for product_predictions in results_iterator:
            raw_predictions = [
                (detokenize_smiles(prediction.text), math.exp(prediction.score))
                for prediction in product_predictions
            ]
            processed_predictions = self._process_raw_predictions(raw_predictions)

            yield raw_predictions, processed_predictions

    def _process_raw_predictions(
        self, raw_predictions: Iterable[RawPrediction]
    ) -> List[ProcessedPrediction]:
        """
        Process the results for one prediction (i.e. one set of precursors).

        Args:
            raw_predictions: tuple containing the raw predicted SMILES (not necessarily
                valid or canonical), and the associated confidence. Is of size ``topn``.

        Returns:
            Lists of tuples, each containing a list of products (usually 1), and
                the confidence. The returned list may be smaller than ``topn``,
                if there were invalid predictions or if some of them were collapsed.
        """
        collapsed_predictions = self.prediction_collapser.collapse_predictions(
            raw_predictions
        )

        results = []

        for smiles, confidence in collapsed_predictions:
            # Note: if the fragment bond is a dot, multiple products will be returned
            products = multicomponent_smiles_to_list(
                smiles, fragment_bond=self.fragment_bond
            )
            if self.canonicalize_output:
                try:
                    products = [canonicalize_smiles(product) for product in products]
                except InvalidSmiles:
                    continue

            if products:
                results.append((products, confidence))

        # sort the results by aggregated confidence
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def prepare_precursors(
        self, precursors_lists: Iterable[List[str]]
    ) -> Iterator[str]:
        """
        Get tokenized inputs for the provided sets of precursors.

        If a canonicalization fails, the corresponding SMILES string will be used as such.

        Returns:
            Iterator over a tokenized SMILES strings (one for each set of precursors).
        """

        for precursors in precursors_lists:
            if self.canonicalize_input:
                precursors = [maybe_canonicalize(smiles) for smiles in precursors]
            if self.sort_input:
                precursors = sorted(precursors)
            multicomponent_smiles = list_to_multicomponent_smiles(
                precursors, self.fragment_bond
            )
            yield tokenize_smiles(multicomponent_smiles)

    @staticmethod
    def _wrap_into_dicts(predictions: Iterable[Any]) -> Iterator[Dict[str, Any]]:
        """Put predictions (raw or processed) into the expected dictionary format."""
        for smiles, confidence in predictions:
            yield {"smiles": smiles, "confidence": confidence}
