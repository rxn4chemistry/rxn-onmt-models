from rxn.onmt_models.prediction_collapser import PredictionCollapser


def test_canonical_collapsing() -> None:
    pc = PredictionCollapser(False)

    raw_predictions = [
        ("CC(O)", 0.4),
        ("CCCO", 0.3),
        ("CCO", 0.2),
        ("CO", 0.04),
        ("C(C)CO", 0.02),
        ("C(C)O", 0.01),
    ]

    # We expect a few predictions to collapse.
    # Note that not the canonical SMILES is kept, but the first suggestion
    expected = [
        ("CC(O)", 0.4 + 0.2 + 0.01),
        ("CCCO", 0.3 + 0.02),
        ("CO", 0.04),
    ]

    assert list(pc.collapse_predictions(raw_predictions)) == expected


def test_hypervalent_smiles() -> None:
    # Default canonicalization doesn't work on these, but the collapser accepts them
    pc = PredictionCollapser(False)

    raw_predictions = [
        ("CF(C)", 0.4),
        ("CFC", 0.3),
    ]

    expected = [
        ("CF(C)", 0.4 + 0.3),
    ]

    assert list(pc.collapse_predictions(raw_predictions)) == expected


def test_invalid_smiles_are_not_removed() -> None:
    pc = PredictionCollapser(False)

    # No change happens here, invalid SMILES are just kept
    raw_predictions = [
        ("CCO", 0.4),
        ("invalid", 0.3),
        ("invalid2", 0.2),
    ]

    assert list(pc.collapse_predictions(raw_predictions)) == raw_predictions


def test_collapse_inchi() -> None:
    # The two following compounds are identical when including the extended InChI,
    # check, but now when canonicalizing them:
    # C1C(=O)CC(=O)CC1
    # C1C(=O)C=C(O)CC1
    #
    # We also include one SMILES augmentation each (that will be collapsed via canonicalization):
    # C1C(=O)CCCC1=O
    # OC1=CC(=O)CCC1

    # No change happens here, invalid SMILES are just kept
    raw_predictions = [
        ("C1C(=O)CC(=O)CC1", 0.3),
        ("C1C(=O)C=C(O)CC1", 0.2),
        ("C1C(=O)CCCC1=O", 0.1),
        ("OC1=CC(=O)CCC1", 0.05),
        ("OC1=CCC(=O)CC1", 0.02),
    ]

    # 1) no InChI collapsing
    pc = PredictionCollapser(collapse_inchi=False)
    expected_no_inchi_collapsing = [
        ("C1C(=O)CC(=O)CC1", 0.3 + 0.1),
        ("C1C(=O)C=C(O)CC1", 0.2 + 0.05),
        ("OC1=CCC(=O)CC1", 0.02),
    ]
    assert (
        list(pc.collapse_predictions(raw_predictions)) == expected_no_inchi_collapsing
    )

    # 2) no InChI collapsing
    pc = PredictionCollapser(collapse_inchi=True)
    expected_with_inchi_collapsing = [
        ("C1C(=O)CC(=O)CC1", 0.3 + 0.2 + 0.1 + 0.05),
        ("OC1=CCC(=O)CC1", 0.02),
    ]
    assert (
        list(pc.collapse_predictions(raw_predictions)) == expected_with_inchi_collapsing
    )


def test_on_multicomponent_smiles() -> None:
    pc = PredictionCollapser(True)

    # Both first predictions should be merged - because of canonicalization
    # for the salt, and InChI for the other compound
    raw_predictions = [
        ("[Cl-]~[Na+].C1C(=O)CC(=O)CC1", 0.3),
        ("C1C(=O)C=C(O)CC1.[Na+]~[Cl-]", 0.2),
        ("[Na+]~[Cl-]", 0.1),
    ]
    expected = [
        ("[Cl-]~[Na+].C1C(=O)CC(=O)CC1", 0.3 + 0.2),
        ("[Na+]~[Cl-]", 0.1),
    ]
    assert list(pc.collapse_predictions(raw_predictions)) == expected


def test_on_reaction_smiles() -> None:
    pc = PredictionCollapser(True)

    # Both first predictions should be merged - because of canonicalization
    # for the salt, and InChI for the other compound
    raw_predictions = [
        ("[Cl-]~[Na+].CC>>C1C(=O)CC(=O)CC1", 0.3),
        ("CC.[Na+]~[Cl-]>>C1C(=O)C=C(O)CC1", 0.2),
        ("CC.[Na+].[Cl-]>>C1C(=O)C=C(O)CC1", 0.1),  # note the difference: no tilde
    ]
    expected = [
        ("[Cl-]~[Na+].CC>>C1C(=O)CC(=O)CC1", 0.3 + 0.2),
        ("CC.[Na+].[Cl-]>>C1C(=O)C=C(O)CC1", 0.1),
    ]
    assert list(pc.collapse_predictions(raw_predictions)) == expected
