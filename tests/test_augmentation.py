import random

from rxn.utilities.files import (
    dump_list_to_file,
    load_list_from_file,
    named_temporary_path,
)

from rxn.onmt_models.augmentation import augment_translation_dataset


def test_augment_partial_reactions() -> None:
    random.seed(42)
    with named_temporary_path() as src_in, named_temporary_path() as src_out, named_temporary_path() as tgt_in, named_temporary_path() as tgt_out:
        dump_list_to_file(["C1NO1.CC(N)C>>", "C.O~N>>CNO", "C.O.N>> |f:1.2|"], src_in)
        dump_list_to_file(["CNC1C(CC)C1>>CN(C)c1cccc(CC)c1", ">>", "CC.O>>CCO"], tgt_in)

        augment_translation_dataset(
            src_in=src_in,
            src_out=src_out,
            tgt_in=tgt_in,
            tgt_out=tgt_out,
            n_augmentations=3,
        )

        # The src has been augmented
        assert load_list_from_file(src_out) == [
            "CC(N)C.N1CO1>>",
            "CC(C)N.O1NC1>>",
            "NC(C)C.N1CO1>>",
            "O~N.C>>CNO",
            "C.O~N>>N(O)C",
            "C.O~N>>CNO",
            "C.O.N>> |f:1.2|",
            "N.O.C>> |f:0.1|",
            "N.O.C>> |f:0.1|",
        ]

        # The tgt just duplicated the lines
        assert load_list_from_file(tgt_out) == [
            "CNC1C(CC)C1>>CN(C)c1cccc(CC)c1",
            "CNC1C(CC)C1>>CN(C)c1cccc(CC)c1",
            "CNC1C(CC)C1>>CN(C)c1cccc(CC)c1",
            ">>",
            ">>",
            ">>",
            "CC.O>>CCO",
            "CC.O>>CCO",
            "CC.O>>CCO",
        ]


def test_augment_compounds() -> None:
    random.seed(42)
    with named_temporary_path() as src_in, named_temporary_path() as src_out, named_temporary_path() as tgt_in, named_temporary_path() as tgt_out:
        dump_list_to_file(["C1NO1.CC(N)C", "C.O~N", "C1C(O)C(=O)C1"], src_in)
        dump_list_to_file(["CN(C)c1cccc(CC)c1", "COCO", "CCO"], tgt_in)

        augment_translation_dataset(
            src_in=src_in,
            src_out=src_out,
            tgt_in=tgt_in,
            tgt_out=tgt_out,
            n_augmentations=3,
        )

        # The src has been augmented
        assert load_list_from_file(src_out) == [
            "CC(N)C.N1CO1",
            "CC(C)N.O1NC1",
            "NC(C)C.N1CO1",
            "O~N.C",
            "O~N.C",
            "C.N~O",
            "C1C(=O)C(O)C1",
            "OC1CCC1=O",
            "OC1C(=O)CC1",
        ]

        # The tgt just duplicated the lines
        assert load_list_from_file(tgt_out) == [
            "CN(C)c1cccc(CC)c1",
            "CN(C)c1cccc(CC)c1",
            "CN(C)c1cccc(CC)c1",
            "COCO",
            "COCO",
            "COCO",
            "CCO",
            "CCO",
            "CCO",
        ]


def test_on_tokenized_input() -> None:
    random.seed(42)
    with named_temporary_path() as src_in, named_temporary_path() as src_out, named_temporary_path() as tgt_in, named_temporary_path() as tgt_out:
        dump_list_to_file(["O . N . C C 1 C O N 1 C", "C . O ~ N"], src_in)
        dump_list_to_file(["N 1 C C ( N ) Br", ">> C O C O"], tgt_in)

        augment_translation_dataset(
            src_in=src_in,
            src_out=src_out,
            tgt_in=tgt_in,
            tgt_out=tgt_out,
            n_augmentations=3,
        )

        # The src has been augmented
        assert load_list_from_file(src_out) == [
            "C C 1 N ( C ) O C 1 . N . O",
            "O . N . C N 1 O C C 1 C",
            "N . C N 1 C ( C ) C O 1 . O",
            "C . O ~ N",
            "N ~ O . C",
            "C . N ~ O",
        ]

        # The tgt just duplicated the lines
        assert load_list_from_file(tgt_out) == [
            "N 1 C C ( N ) Br",
            "N 1 C C ( N ) Br",
            "N 1 C C ( N ) Br",
            ">> C O C O",
            ">> C O C O",
            ">> C O C O",
        ]


def test_keep_original() -> None:
    random.seed(42)
    with named_temporary_path() as src_in, named_temporary_path() as src_out, named_temporary_path() as tgt_in, named_temporary_path() as tgt_out:
        dump_list_to_file(["O.N.CC1CON1C", "C.O~N"], src_in)
        dump_list_to_file(["N1CC(N)Br", ">>COCO"], tgt_in)

        augment_translation_dataset(
            src_in=src_in,
            src_out=src_out,
            tgt_in=tgt_in,
            tgt_out=tgt_out,
            n_augmentations=3,
            keep_original=True,
        )

        # The src has been augmented, the original is given first
        assert load_list_from_file(src_out) == [
            "O.N.CC1CON1C",
            "CC1N(C)OC1.N.O",
            "O.N.CN1OCC1C",
            "N.CN1C(C)CO1.O",
            "C.O~N",
            "C.O~N",
            "N~O.C",
            "C.N~O",
        ]

        # The tgt just duplicated the lines
        assert load_list_from_file(tgt_out) == [
            "N1CC(N)Br",
            "N1CC(N)Br",
            "N1CC(N)Br",
            "N1CC(N)Br",
            ">>COCO",
            ">>COCO",
            ">>COCO",
            ">>COCO",
        ]
