# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2021
# ALL RIGHTS RESERVED

from argparse import Namespace
from typing import List, Iterable, Union, Optional, Any

from .internal_translation_utils import TranslationResult, RawTranslator, get_onmt_opt


class Translator:
    """
    Wraps the OpenNMT translation functionality into a class.
    """

    def __init__(self, opt: Namespace):
        """
        Should not be called directly as implementation may change; call the
        classmethods from_model_path or from_opt instead.

        Args:
            opt: model options.
        """
        self.onmt_translator = RawTranslator(opt=opt)

    def translate_single(self, sentence: str) -> str:
        """
        Translate one single sentence.
        """
        translations = self.translate_sentences([sentence])
        assert len(translations) == 1
        return translations[0]

    def translate_sentences(self, sentences: List[str]) -> List[str]:
        """
        Translate multiple sentences.
        """
        translations = self.translate_multiple_with_scores(sentences)
        return [t[0].text for t in translations]

    def translate_multiple_with_scores(self,
                                       sentences: List[str],
                                       n_best: Optional[int] = None
                                       ) -> List[List[TranslationResult]]:
        """
        Translate multiple sentences.

        Args:
            sentences: Sentences to translate.
            n_best: if provided, will overwrite the number of predictions to make.
        """
        additional_opt_kwargs = {}
        if n_best is not None:
            additional_opt_kwargs['n_best'] = n_best

        translations = self.onmt_translator.translate_sentences_with_onmt(
            sentences, **additional_opt_kwargs
        )

        return translations

    @classmethod
    def from_model_path(cls, model_path: Union[str, Iterable[str]], **kwargs: Any):
        """
        Create a Translator instance from the model path(s).

        Args:
            model_path: path to the translation model file(s).
                If multiple are given, will be an ensemble model.
            kwargs: Additional values to be parsed for instantiating the translator,
                such as n_best, beam_size, max_length, etc.
        """
        if isinstance(model_path, str):
            model_path = [model_path]
        opt = get_onmt_opt(translation_model=list(model_path), **kwargs)
        return cls(opt=opt)

    @classmethod
    def from_opt(cls, opt: Namespace):
        """
        Create a Translator instance from the opt arguments.

        Args:
            opt: model options.
        """
        return cls(opt=opt)
