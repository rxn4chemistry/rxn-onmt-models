from typing import List, Iterable, Union

from .internal_translation_utils import get_onmt_opt, TranslationResult, RawTranslator


class Translator:
    """
    Wraps the OpenNMT translation functionality into a class.
    """

    def __init__(self, model: Union[str, Iterable[str]]):
        """
        Args:
            model: path to the translation model file(s).
                If multiple are given, will be an ensemble model.
        """

        if isinstance(model, str):
            model = [model]
        self.model = list(model)

        self.onmt_translator = RawTranslator(self.model)

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
                                       n_best=1) -> List[List[TranslationResult]]:
        onmt_opt = get_onmt_opt(translation_model=self.model, n_best=n_best)

        translations = self.onmt_translator.translate_sentences_with_onmt(onmt_opt, sentences)

        return translations
