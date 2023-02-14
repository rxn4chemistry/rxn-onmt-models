from pathlib import Path

from rxn.utilities.files import PathLike


class MetricsFiles:
    def __init__(
        self,
        directory: PathLike,
        gt_src: str = "gt_src.txt",
        gt_tgt: str = "gt_tgt.txt",
        predicted: str = "pred.txt",
        predicted_canonical: str = "predicted_canonical.txt",
    ):
        self.directory = Path(directory)
        self.log_file = self.directory / "log.txt"
        self.metrics_file = self.directory / "metrics.json"
        self.gt_src = self.directory / gt_src
        self.gt_tgt = self.directory / gt_tgt
        self.predicted = self.directory / predicted
        self.predicted_canonical = self.directory / predicted_canonical


class RetroFiles(MetricsFiles):
    """
    Class holding the locations of the files to write to or to read from for
    the evaluation of retro metrics.
    """

    _REORDERED_FILE_EXTENSION = ".reordered"

    def __init__(self, directory: PathLike):
        super().__init__(
            directory=directory,
            gt_src="gt_products.txt",
            gt_tgt="gt_precursors.txt",
            predicted="predicted_precursors.txt",
            predicted_canonical="predicted_precursors_canonical.txt",
        )
        self.class_token_products = self.directory / "class_token_products.txt"
        self.class_token_precursors = self.directory / "class_token_precursors.txt"
        self.predicted_precursors_log_probs = (
            self.directory / "predicted_precursors.txt.tokenized_log_probs"
        )
        self.predicted_products = self.directory / "predicted_products.txt"
        self.predicted_products_canonical = (
            self.directory / "predicted_products_canonical.txt"
        )
        self.predicted_products_log_probs = (
            self.directory / "predicted_products.txt.tokenized_log_probs"
        )
        self.predicted_rxn_canonical = self.directory / "predicted_rxn_canonical.txt"
        self.predicted_classes = self.directory / "predicted_classes.txt"

    @staticmethod
    def reordered(path: PathLike) -> Path:
        """Add the reordered path extension."""
        return Path(str(path) + RetroFiles._REORDERED_FILE_EXTENSION)


class ForwardFiles(MetricsFiles):
    """
    Class holding the locations of the files to write to or to read from for
    the evaluation of forward metrics.
    """

    def __init__(self, directory: PathLike):
        super().__init__(
            directory=directory,
            gt_src="gt_precursors.txt",
            gt_tgt="gt_products.txt",
            predicted="predicted_products.txt",
            predicted_canonical="predicted_products_canonical.txt",
        )


class ContextFiles(MetricsFiles):
    """
    Class holding the locations of the files to write to or to read from for
    the evaluation of context metrics.
    """

    def __init__(self, directory: PathLike):
        super().__init__(
            directory=directory,
            predicted="predicted_context.txt",
            predicted_canonical="predicted_context_canonical.txt",
        )
