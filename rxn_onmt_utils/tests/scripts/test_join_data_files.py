import os.path

from rxn.utilities.files import (
    dump_list_to_file,
    load_list_from_file,
    named_temporary_path,
)

from rxn_onmt_utils.scripts.join_data_files import join_data_files


def test_join_data_files():
    # Creation of a temporary directory to dump output files
    with named_temporary_path() as temporary_path:
        chunk_dir_0 = temporary_path / "splitted_data" / "chunk_0"
        chunk_dir_0.mkdir(parents=True, exist_ok=True)
        chunk_dir_1 = temporary_path / "splitted_data" / "chunk_1"
        chunk_dir_1.mkdir(parents=True, exist_ok=True)
        chunk_dir_2 = temporary_path / "splitted_data" / "chunk_2"
        chunk_dir_2.mkdir(parents=True, exist_ok=True)

        dump_list_to_file(["A", "B"], chunk_dir_0 / "gt_products.txt")
        dump_list_to_file(["A.B.C", "D.E.F"], chunk_dir_0 / "gt_precursors.txt")
        dump_list_to_file(["C", "D"], chunk_dir_1 / "gt_products.txt")
        dump_list_to_file(["G.H.I", "L.M.N"], chunk_dir_1 / "gt_precursors.txt")
        dump_list_to_file(["E"], chunk_dir_2 / "gt_products.txt")
        dump_list_to_file(["O.P.Q"], chunk_dir_2 / "gt_precursors.txt")

        join_data_files(
            input_dir=temporary_path / "splitted_data",
            output_dir=temporary_path,
        )

        assert os.path.exists(temporary_path / "gt_products.txt")
        assert os.path.exists(temporary_path / "gt_precursors.txt")

        assert load_list_from_file(temporary_path / "gt_products.txt") == [
            "A",
            "B",
            "C",
            "D",
            "E",
        ]
        assert load_list_from_file(temporary_path / "gt_precursors.txt") == [
            "A.B.C",
            "D.E.F",
            "G.H.I",
            "L.M.N",
            "O.P.Q",
        ]


def test_join_data_files_skipped(caplog):
    # Creation of a temporary directory to dump output files
    with named_temporary_path() as temporary_path:
        chunk_dir_0 = temporary_path / "splitted_data" / "chunk_0"
        chunk_dir_0.mkdir(parents=True, exist_ok=True)
        chunk_dir_1 = temporary_path / "splitted_data" / "chunk_1"
        chunk_dir_1.mkdir(parents=True, exist_ok=True)
        chunk_dir_2 = temporary_path / "splitted_data" / "chunk_2"
        chunk_dir_2.mkdir(parents=True, exist_ok=True)

        dump_list_to_file(["A", "B"], chunk_dir_0 / "gt_products.txt")
        dump_list_to_file(["A.B.C", "D.E.F"], chunk_dir_0 / "gt_precursors.txt")
        dump_list_to_file(["C", "D"], chunk_dir_1 / "gt_products.txt")
        dump_list_to_file(["G.H.I", "L.M.N"], chunk_dir_1 / "gt_precursors.txt")
        dump_list_to_file(["E"], chunk_dir_2 / "gt_products.txt")
        dump_list_to_file(["O.P.Q"], chunk_dir_2 / "gt_precursors.txt")

        # Adding a file in only one of the directories
        dump_list_to_file(["A", "B"], chunk_dir_0 / "a.txt")

        join_data_files(
            input_dir=temporary_path / "splitted_data",
            output_dir=temporary_path,
        )

        assert [r.msg for r in caplog.records] == [
            f"The file '{chunk_dir_1 / 'a.txt'}' does not exist. Not joining",
            f"The file '{chunk_dir_2 / 'a.txt'}' does not exist. Not joining",
        ]
