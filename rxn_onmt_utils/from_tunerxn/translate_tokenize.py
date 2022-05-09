#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
from typing import Tuple

import pandas as pd
import rxn_reaction_preprocessing as rrp


def translate_tokenize(input_output_pairs: Tuple[str, ...], model_task: str) -> None:
    """Tokenize before translation

    Args:
        input_output_pairs:  Paths to the input and output files in the form <input_a> <output_a> <input_b> <output_b> [...].
        model_task: 'forward' or 'retro'
    """

    if len(input_output_pairs) % 2 != 0:
        print()
        raise SystemExit(
            f'Files must be supplied as input output pairs in the form <input_a> <output_a> <input_b> <output_b> [...]:\n{input_output_pairs}'
        )

    # Tokenize the reactions
    tokenizer = rrp.SmilesTokenizer()

    for input, output in zip(input_output_pairs[::2], input_output_pairs[1::2]):
        df = pd.read_csv(input, lineterminator='\n')

        if 'rxn' not in df.columns:
            raise SystemExit(f'The following file does not contain an rxn column:\n{input}')

        if model_task == 'forward':
            split_to_keep = 0
        elif model_task == 'retro':
            split_to_keep = 1
        else:
            raise ValueError(f'model_task should be "forward" or "retro" (actual: "{model_task}")')

        df['for_translation'] = df.rxn.str.split('>>').str[split_to_keep]
        df.for_translation = df.for_translation.apply(tokenizer.tokenize)
        df[['for_translation']].to_csv(f'{output}', header=False, index=False)
