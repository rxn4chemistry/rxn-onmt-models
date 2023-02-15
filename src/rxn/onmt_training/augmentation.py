"""
Augmentation of source and target files in the context of RXN translation models.
"""

import itertools
from pathlib import Path
from typing import Callable, List, Optional

from rxn.chemutils.smiles_augmenter import SmilesAugmenter
from rxn.chemutils.smiles_randomization import randomize_smiles_rotated
from rxn.chemutils.tokenization import (
    detokenize_smiles,
    file_is_tokenized,
    tokenize_smiles,
)
from rxn.utilities.files import (
    dump_list_to_file,
    iterate_lines_from_file,
    raise_if_paths_are_identical,
)


def augment_translation_dataset(
    *,
    src_in: Path,
    src_out: Path,
    tgt_in: Path,
    tgt_out: Path,
    n_augmentations: int,
    keep_original: bool = False,
    augmenter: Optional[SmilesAugmenter] = None,
) -> None:
    """
    Augment an RXN translation dataset with randomization and shuffling of the source.

    The target will not be modified, but its elements will be duplicated if more
    than one augmentation is required.
    The first argument, `*`, forces keyword arguments only (Python Cookbook 7.2).

    Notes (Alain, Nov 2022):
      - In the future, we may want to convert this functionality to a class.

    Args:
        src_in: source file to augment, in TXT format (tokenized or not).
        src_out: augmented source file.
        tgt_in: target file to augment (i.e. duplicate its samples).
        tgt_out: augmented target file.
        n_augmentations: number of augmentations per input line.
        keep_original: whether to keep the original sample in the output.
        augmenter: augmenter instance. Defaults to a rotated augmentation of
            the compound SMILES.
    """

    if augmenter is None:
        augmenter = SmilesAugmenter(
            augmentation_fn=randomize_smiles_rotated,
            augmentation_probability=1.0,
            shuffle=True,
        )

    augment_src(
        src_in=src_in,
        src_out=src_out,
        n_augmentations=n_augmentations,
        keep_original=keep_original,
        augmenter=augmenter,
    )
    augment_tgt(
        tgt_in=tgt_in,
        tgt_out=tgt_out,
        n_augmentations=n_augmentations,
        keep_original=keep_original,
    )


def augment_src(
    src_in: Path,
    src_out: Path,
    n_augmentations: int,
    keep_original: bool,
    augmenter: SmilesAugmenter,
) -> None:
    """
    Augment a source file.

    Args:
        src_in: source file to augment, in TXT format (tokenized or not).
        src_out: augmented source file.
        n_augmentations: number of augmentations per input line.
        keep_original: whether to keep the original sample in the output.
        augmenter: augmenter instance.
    """
    raise_if_paths_are_identical(src_in, src_out)

    augment = _augment_callback(
        augmenter,
        is_tokenized=file_is_tokenized(src_in),
        n_augmentations=n_augmentations,
        keep_original=keep_original,
    )

    src_smiles_in = iterate_lines_from_file(src_in)

    src_smiles_out = (
        augmented for smiles in src_smiles_in for augmented in augment(smiles)
    )

    dump_list_to_file(src_smiles_out, src_out)


def augment_tgt(
    tgt_in: Path,
    tgt_out: Path,
    n_augmentations: int,
    keep_original: bool,
) -> None:
    """
    Augment a target file (i.e., duplicate its elements).

    Args:
        tgt_in: target file to augment (i.e. duplicate its samples).
        tgt_out: augmented target file.
        n_augmentations: number of augmentations per input line.
        keep_original: whether to keep the original sample in the output.
    """
    raise_if_paths_are_identical(tgt_in, tgt_out)

    if keep_original:
        # If we want to keep the original, we will need one more copy.
        n_augmentations += 1

    tgt_smiles_in = iterate_lines_from_file(tgt_in)
    # Repeat the target in the output: ABC -> AAAABBBBCCCC
    tgt_smiles_out = itertools.chain.from_iterable(
        itertools.repeat(smiles, n_augmentations) for smiles in tgt_smiles_in
    )
    dump_list_to_file(tgt_smiles_out, tgt_out)


def _augment_callback(
    augmenter: SmilesAugmenter,
    is_tokenized: bool,
    n_augmentations: int,
    keep_original: bool,
) -> Callable[[str], List[str]]:
    """
    Generate the callback to do the augmentation, also depending on whether the
    file is tokenized and whether to keep the original samples.
    """

    def fn(smiles: str) -> List[str]:
        # for augmentation, we first need to detokenize if needed
        if is_tokenized:
            smiles = detokenize_smiles(smiles)

        augmented = augmenter.augment(smiles, n_augmentations)

        # insert original in first place, if necessary
        if keep_original:
            augmented = [smiles] + augmented

        # tokenize (if needed)
        if is_tokenized:
            augmented = [tokenize_smiles(s) for s in augmented]

        return augmented

    return fn
