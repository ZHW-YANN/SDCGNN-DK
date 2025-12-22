from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
import numpy as np
from monty.serialization import loadfn
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.core import Element, Structure
from data_loader.graph import Converter, StructureGraph, StructureGraphFixedRadius
from data_loader.graph import GaussianDistance


class CrystalGraph(StructureGraphFixedRadius):
    """
    Convert a crystal into a graph with z as atomic feature and distance as bond feature
    one can optionally include state features
    """

    def __init__(
        self,
        nn_strategy: str | NearNeighbors = "MinimumDistanceNNAll",
        atom_converter: Converter | None = None,
        bond_converter: Converter | None = None,
        cutoff: float = 5.0,
    ):
        """
        Convert the structure into crystal graph
        Args:
            nn_strategy (str): NearNeighbor strategy
            atom_converter (Converter): atom features converter
            bond_converter (Converter): bond features converter
            cutoff (float): cutoff radius
        """
        self.cutoff = cutoff
        super().__init__(
            nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter, cutoff=self.cutoff
        )



struture = Structure.from_file('../data_test/cif/icsd_14.cif')

# cg = CrystalGraph(cutoff=8, bond_converter=GaussianDistance(np.linspace(0, 5, 100), 0.5))
cg = CrystalGraph(cutoff=8)
graph = cg.convert(struture)
print(graph)

