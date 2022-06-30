#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
import itertools
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rxn.chemutils.conversion import canonicalize_smiles, smiles_to_inchi
from rxn.chemutils.tokenization import detokenize_smiles

INVALID_PRODUCT = "[INVALID]"


class SinglePrediction:
    def __init__(self, tokenized_smiles: str, confidence: float):
        self.confidence = confidence
        self.smiles: Optional[str]
        self.inchi: Optional[str]
        try:
            self.smiles = canonicalize_smiles(detokenize_smiles(tokenized_smiles))
            self.inchi = smiles_to_inchi(self.smiles)
        except Exception:
            self.smiles = None
            self.inchi = None


def output_indices(df: pd.DataFrame, output_tag: str) -> List[int]:
    indices = []
    for column_name in df.columns:
        if f"{output_tag}_" in column_name:
            indices.append(int(column_name.split("_")[1]))
    return sorted(indices)


def postprocess_predictions_and_confidences(
    record_from_df: Dict[str, Any], indices: List[int], output_tag: str
) -> List[Tuple[Optional[str], float]]:
    # parse the predictions and associated confidences
    vals = [
        (record_from_df[f"{output_tag}_{i}"], record_from_df[f"confidence_{i}"])
        for i in indices
    ]

    # Create SinglePrediction instances from it, and keep only the valid ones
    predictions = [
        SinglePrediction(tokenized_smiles, confidence)
        for tokenized_smiles, confidence in vals
    ]
    valid_predictions = [
        prediction for prediction in predictions if prediction.smiles is not None
    ]

    # Collapse to single InChI values. In this case only the first SMILES is kept.
    dict_of_unique_inchis = {}
    for prediction in valid_predictions:
        if prediction.inchi not in dict_of_unique_inchis:
            dict_of_unique_inchis[prediction.inchi] = prediction
        else:
            dict_of_unique_inchis[prediction.inchi].confidence += prediction.confidence

    # Put back in a list and sort by decreasing confidence
    collapsed_predictions = list(dict_of_unique_inchis.values())
    collapsed_predictions.sort(key=lambda x: x.confidence, reverse=True)
    return [
        (prediction.smiles, prediction.confidence)
        for prediction in collapsed_predictions
    ]


def translate_postprocess(
    reactant_input: str, predictions: str, combined_output: str, model_task: str
) -> None:
    """Combine and postprocess the results.

    Args:
        reactant_input: original input file (one reaction per line).
        predictions: tokenized model predictions.
        combined_output: where to save the output file.
        model_task: 'forward' or 'retro'.
    """
    if model_task == "forward":
        input_tag = "precursors"
        output_tag = "product"
    elif model_task == "retro":
        input_tag = "product"
        output_tag = "precursors"
    else:
        raise ValueError(
            f'model_task should be "forward" or "retro" (actual: "{model_task}")'
        )

    input_df = pd.read_csv(reactant_input)
    input_df[input_tag] = input_df["rxn"].str.replace(">>", "")
    input_df.drop(["rxn"], axis=1, inplace=True)
    input_df.reset_index(drop=True, inplace=True)

    output_df = pd.read_csv(predictions)
    indices = output_indices(output_df, output_tag=output_tag)

    # Update the pandas DataFrame by detokenizing, canonicalizing, and collapsing
    # for identical InChIs.
    records = output_df.to_dict("records")
    for r in records:
        vs = postprocess_predictions_and_confidences(r, indices, output_tag=output_tag)
        for i, (smiles, confidence) in itertools.zip_longest(
            indices, vs, fillvalue=(INVALID_PRODUCT, 0.0)
        ):
            r[f"{output_tag}_{i}"] = smiles
            r[f"confidence_{i}"] = confidence
    output_df = pd.DataFrame(records)

    # Concatenate and write to CSV
    output_df.drop([input_tag], axis=1, inplace=True)
    combined = pd.concat([input_df, output_df], axis=1)
    combined.to_csv(combined_output, index=False)
