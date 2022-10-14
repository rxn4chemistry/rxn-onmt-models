# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2020
# ALL RIGHTS RESERVED
import re

import rxn.reaction_preprocessing as rrp
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def translate_preprocess(
    input_file_path: str,
    output_file_path: str,
    fragment_bond: str = ".",
    max_reactants: int = 10,
    max_agents: int = 0,
    max_products: int = 1,
    min_reactants: int = 2,
    min_agents: int = 0,
    min_products: int = 1,
    max_reactants_tokens: int = 300,
    max_agents_tokens: int = 0,
    max_products_tokens: int = 200,
    max_absolute_formal_charge: int = 2,
) -> None:
    """The entry point for this cli script.

    Args:
        input_file_path:  The input file path (one reaction SMARTS per line).
        output_file_path: The output file path.
        fragment_bond: The fragment bond character.
        max_reactants: The maximum number of reactants.
        max_agents: The maximum number of agents.
        max_products: The maximum number of products.
        min_reactants: The minimum number of reactants.
        min_agents: The minimum number of agents.
        min_products: The minimum number of products.
        max_reactants_tokens: The maximum number of reactants tokens.
        max_agents_tokens: The maximum number of agents tokens.
        max_products_tokens: The maximum number of products tokens.
        max_absolute_formal_charge: The maximum absolute formal charge.
    """

    # This is for the special SMILES extension where agents are separated by pipe.
    def clean_func(rxn: str) -> str:
        return re.sub(r"(?<=\[)([0-9]+)(?=[A-Za-z])", "", rxn)  # Remove isotopes

    # This is the function that is applied to each reaction.
    def apply_func(reaction: rrp.Reaction) -> rrp.Reaction:
        # Move agents to reactants
        reaction.remove_none()
        reaction.reactants.extend(reaction.agents)
        reaction.agents = []

        # Remove products that are also reactants
        reaction.remove_precursors_from_products()

        return reaction.sort()

    # Create a instance of the mixed reaciton filter with default values.
    # Make arguments for all properties in script
    mrf = rrp.MixedReactionFilter(
        max_reactants=max_reactants,
        max_agents=max_agents,
        max_products=max_products,
        min_reactants=min_reactants,
        min_agents=min_agents,
        min_products=min_products,
        max_reactants_tokens=max_reactants_tokens,
        max_agents_tokens=max_agents_tokens,
        max_products_tokens=max_products_tokens,
        max_absolute_formal_charge=max_absolute_formal_charge,
    )

    pp = rrp.Preprocessor.read_csv(input_file_path, "rxn", fragment_bond=fragment_bond)

    # In a first step, let's clean the data using the cleaning function
    # defined above
    pp.df.rxn = pp.df.rxn.apply(clean_func)

    # Apply the function above to all reactions, the remove_duplicate_molecules argument
    # is set to true to remove duplicate molecules within each reaction part
    pp.apply(apply_func, remove_duplicate_molecules=True)

    # Apply the mixed reaction filter instance defined above, enable verbose mode
    pp.filter(mrf, True)

    # After dropping invalid columns, display stats again (as an example)
    pp.df.rxn.to_csv(output_file_path)
