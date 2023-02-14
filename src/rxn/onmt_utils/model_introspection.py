from argparse import Namespace
from typing import Any, Dict, List

import torch
from onmt.inputters.text_dataset import TextMultiField
from rxn.utilities.files import PathLike


def get_model_vocab(model_path: PathLike) -> List[str]:
    """
    Get the vocabulary from a model checkpoint.

    Args:
        model_path: model checkpoint, such as `model_step_100000.pt`.
    """
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    vocab = checkpoint["vocab"]
    return _torch_vocab_to_list(vocab)


def get_preprocessed_vocab(vocab_path: PathLike) -> List[str]:
    """
    Get the vocabulary from the file saved by OpenNMT during preprocessing.

    Args:
        vocab_path: vocab file, such as `preprocessed.vocab.pt`.
    """
    vocab = torch.load(vocab_path)
    return _torch_vocab_to_list(vocab)


def model_vocab_is_compatible(model_pt: PathLike, vocab_pt: PathLike) -> bool:
    """
    Determine whether the vocabulary contained in a model checkpoint contains
    all the necessary tokens from a vocab file.

    Args:
        model_pt: model checkpoint, such as `model_step_100000.pt`.
        vocab_pt: vocab file, such as `preprocessed.vocab.pt`.
    """
    model_vocab = set(get_model_vocab(model_pt))
    data_vocab = set(get_preprocessed_vocab(vocab_pt))
    return data_vocab.issubset(model_vocab)


def _torch_vocab_to_list(vocab: Dict[str, Any]) -> List[str]:
    src_vocab = _multifield_vocab_to_list(vocab["src"])
    tgt_vocab = _multifield_vocab_to_list(vocab["tgt"])
    if src_vocab != tgt_vocab:
        raise RuntimeError("Handling of different src/tgt vocab not implemented")
    return src_vocab


def _multifield_vocab_to_list(multifield: TextMultiField) -> List[str]:
    return multifield.base_field.vocab.itos[:]


def get_model_opt(model_path: PathLike) -> Namespace:
    """
    Get the args ("opt") of rnn_size for the given model checkpoint.

    Args:
        model_path: model checkpoint, such as `model_step_100000.pt`.
    """
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    return checkpoint["opt"]


def get_model_rnn_size(model_path: PathLike) -> int:
    """
    Get the value of rnn_size for the given model checkpoint.

    Args:
        model_path: model checkpoint, such as `model_step_100000.pt`.
    """
    return get_model_opt(model_path).rnn_size


def get_model_dropout(model_path: PathLike) -> float:
    """
    Get the value of the dropout for the given model checkpoint.

    Args:
        model_path: model checkpoint, such as `model_step_100000.pt`.
    """

    # Note: OpenNMT has support for several dropout values (changing during
    # training). We do not support this at the moment.
    dropouts = get_model_opt(model_path).dropout
    if len(dropouts) != 1:
        raise ValueError(f"Expected one dropout value. Actual: {dropouts}")
    return dropouts[0]


def get_model_seed(model_path: PathLike) -> int:
    """
    Get the value of the seed for the given model checkpoint.

    Args:
        model_path: model checkpoint, such as `model_step_100000.pt`.
    """
    return get_model_opt(model_path).seed
