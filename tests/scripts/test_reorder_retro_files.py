import tempfile
from pathlib import Path

from rxn.utilities.files import dump_list_to_file, load_list_from_file

from rxn.onmt_training.rxn_models.metrics_files import RetroFiles
from rxn.onmt_training.scripts.reorder_retro_predictions_class_token import (
    reorder_retro_predictions_class_token,
)


def test_reorder_retro_files() -> None:
    # Creation of a temporary directory to dump output files
    with tempfile.TemporaryDirectory() as temporary_dir:
        """
        In this test we process just one reaction and there will be 4 predictions for each class token used.
        In total 8 predictions given 2 class tokens. The code should reorder the predictions class token-wise:
        by looking at the top1 predictions for each class token, reordering and then looking at the top2
        predictions and reordering. The reordering relies on the values of the negative log likelihood.
        """

        temporary_path = Path(temporary_dir)

        dump_list_to_file(["A"], temporary_path / "gt.txt")
        dump_list_to_file(
            ["1", "2", "3", "4", "5", "6", "7", "8"], temporary_path / "pred.txt"
        )
        dump_list_to_file(
            ["-4.3", "-5.9", "-6.0", "-12.0", "-1.1", "-6.9", "-7.0", "-9.4"],
            temporary_path / "conf.txt",
        )
        dump_list_to_file(
            ["11", "22", "33", "44", "55", "66", "77", "88"],
            temporary_path / "fwd_pred.txt",
        )
        dump_list_to_file(
            ["1.1", "2.2", "3.3", "4.4", "5.5", "6.6", "7.7", "8.8"],
            temporary_path / "class_pred.txt",
        )

        reorder_retro_predictions_class_token(
            ground_truth_file=temporary_path / "gt.txt",
            predictions_file=temporary_path / "pred.txt",
            confidences_file=temporary_path / "conf.txt",
            fwd_predictions_file=temporary_path / "fwd_pred.txt",
            classes_predictions_file=temporary_path / "class_pred.txt",
            n_class_tokens=2,
        )
        assert load_list_from_file(
            RetroFiles.reordered(temporary_path / "pred.txt")
        ) == ["5", "1", "2", "6", "3", "7", "8", "4"]
        assert load_list_from_file(
            RetroFiles.reordered(temporary_path / "conf.txt")
        ) == ["-1.1", "-4.3", "-5.9", "-6.9", "-6.0", "-7.0", "-9.4", "-12.0"]
        assert load_list_from_file(
            RetroFiles.reordered(temporary_path / "fwd_pred.txt")
        ) == ["55", "11", "22", "66", "33", "77", "88", "44"]
        assert load_list_from_file(
            RetroFiles.reordered(temporary_path / "class_pred.txt")
        ) == ["5.5", "1.1", "2.2", "6.6", "3.3", "7.7", "8.8", "4.4"]
