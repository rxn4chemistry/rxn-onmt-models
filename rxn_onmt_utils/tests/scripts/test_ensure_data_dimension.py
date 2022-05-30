import tempfile
from pathlib import Path

import pytest
from rxn_utilities.file_utilities import dump_list_to_file, load_list_from_file

from rxn_onmt_utils.scripts.ensure_data_dimension import ensure_data_dimension


def test_ensure_data_dimension():
    # Creation of a temporary directory to dump output files
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_path = Path(temporary_dir)

        dump_list_to_file(
            ["A.B.C", "D.E.F", "G.H.I", "L.M.N", "O.P.Q"],
            temporary_path / "precursors.txt",
        )
        dump_list_to_file(["A", "B", "C", "D", "E"], temporary_path / "products.txt")

        files = (temporary_path / "precursors.txt", temporary_path / "products.txt")
        ensure_data_dimension(
            txt_files=files, output_dir=temporary_dir, max_dimension=2
        )

        assert temporary_path.exists()
        assert (temporary_path / "chunk_0").exists()
        assert (temporary_path / "chunk_1").exists()
        assert (temporary_path / "chunk_2").exists()

        assert load_list_from_file(temporary_path / "chunk_0" / "products.txt") == [
            "A",
            "B",
        ]
        assert load_list_from_file(temporary_path / "chunk_0" / "precursors.txt") == [
            "A.B.C",
            "D.E.F",
        ]
        assert load_list_from_file(temporary_path / "chunk_1" / "products.txt") == [
            "C",
            "D",
        ]
        assert load_list_from_file(temporary_path / "chunk_1" / "precursors.txt") == [
            "G.H.I",
            "L.M.N",
        ]
        assert load_list_from_file(temporary_path / "chunk_2" / "products.txt") == ["E"]
        assert load_list_from_file(temporary_path / "chunk_2" / "precursors.txt") == [
            "O.P.Q"
        ]


def test_ensure_data_dimension_one_chunk_only():
    # Creation of a temporary directory to dump output files
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_path = Path(temporary_dir)

        dump_list_to_file(
            ["A.B.C", "D.E.F", "G.H.I", "L.M.N", "O.P.Q"],
            temporary_path / "precursors.txt",
        )
        dump_list_to_file(["A", "B", "C", "D", "E"], temporary_path / "products.txt")

        files = (temporary_path / "precursors.txt", temporary_path / "products.txt")
        ensure_data_dimension(
            txt_files=files, output_dir=temporary_dir, max_dimension=6
        )

        assert temporary_path.exists()
        assert (temporary_path / "chunk_0").exists()
        assert not (temporary_path / "splitted_data" / "chunk_1").exists()

        assert load_list_from_file(temporary_path / "chunk_0" / "products.txt") == [
            "A",
            "B",
            "C",
            "D",
            "E",
        ]
        assert load_list_from_file(temporary_path / "chunk_0" / "precursors.txt") == [
            "A.B.C",
            "D.E.F",
            "G.H.I",
            "L.M.N",
            "O.P.Q",
        ]


def test_ensure_data_dimension_different_length():
    # Creation of a temporary directory to dump output files
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_path = Path(temporary_dir)

        dump_list_to_file(
            ["A.B.C", "D.E.F", "G.H.I", "L.M.N"], temporary_path / "file1.txt"
        )
        dump_list_to_file(["A", "B"], temporary_path / "file2.txt")

        files = (temporary_path / "file1.txt", temporary_path / "file2.txt")

        with pytest.raises(ValueError):
            ensure_data_dimension(
                txt_files=files, output_dir=temporary_dir, max_dimension=3
            )
