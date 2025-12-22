"""Abstract classes and utility operations for building graph representations and
data loaders (known as Sequence objects in Keras).
Most users will not need to interact with this module."""
from __future__ import annotations

from abc import abstractmethod
from inspect import signature
from logging import getLogger
from operator import itemgetter

import numpy as np
from monty.json import MSONable
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.core import Structure
from data_loader import local_env
from data_loader.data_utils import get_graphs_within_cutoff
from data_loader.data_utils import expand_1st, to_list


logger = getLogger(__name__)


class Converter(MSONable):
    """
    Base class for atom or bond converter
    """

    def convert(self, d):
        """
        Convert the object d
        Args:
            d (Any): Any object d

        Returns: returned object
        """
        raise NotImplementedError


class StructureGraph(MSONable):
    """
    This is a base class for converting converting structure into graphs or model inputs
    Methods to be implemented are follows:
        1. convert(self, structure)
            This is to convert a structure into a graph dictionary
        2. get_input(self, structure)
            This method convert a structure directly to a model input
        3. get_flat_data(self, graphs, targets)
            This method process graphs and targets pairs and output model input list.
    """

    def __init__(
        self,
        nn_strategy: str | NearNeighbors | None = None,
        atom_converter: Converter | None = None,
        bond_converter: Converter | None = None,
        **kwargs,
    ):
        """

        Args:
            nn_strategy (str or NearNeighbors): NearNeighbor strategy
            atom_converter (Converter): atom converter
            bond_converter (Converter): bond converter
            **kwargs:
        """

        if isinstance(nn_strategy, str):
            strategy = local_env.get(nn_strategy)
            parameters = signature(strategy).parameters
            param_dict = {i: j.default for i, j in parameters.items()}
            for i, j in kwargs.items():
                if i in param_dict:
                    setattr(self, i, j)
                    param_dict.update({i: j})
            self.nn_strategy = strategy(**param_dict)
        elif isinstance(nn_strategy, NearNeighbors):
            self.nn_strategy = nn_strategy
        elif nn_strategy is None:
            self.nn_strategy = None
        else:
            raise RuntimeError("Strategy not valid")

        self.atom_converter = atom_converter or self._get_dummy_converter()
        self.bond_converter = bond_converter or self._get_dummy_converter()

    def convert(self, structure: Structure, state_attributes: list | None = None) -> dict:
        """
        Take a pymatgen structure and convert it to a index-type graph representation
        The graph will have node, distance, index1, index2, where node is a vector of Z number
        of atoms in the structure, index1 and index2 mark the atom indices forming the bond and separated by
        distance.
        For state attributes, you can set structure.state = [[xx, xx]] beforehand or the algorithm would
        take default [[0, 0]]
        Args:
            state_attributes: (list) state attributes
            structure: (pymatgen structure)
            (dictionary)
        """
        state_attributes = (
            state_attributes or getattr(structure, "state", None) or np.array([[0.0, 0.0]], dtype="float32")
        )
        index1 = []
        index2 = []
        bonds = []
        if self.nn_strategy is None:
            raise RuntimeError("NearNeighbor strategy is not provided!")
        for n, neighbors in enumerate(self.nn_strategy.get_all_nn_info(structure)):
            index1.extend([n] * len(neighbors))
            for neighbor in neighbors:
                index2.append(neighbor["site_index"])
                bonds.append(neighbor["weight"])
        atoms = self.get_atom_features(structure)
        if np.size(np.unique(index1)) < len(atoms):
            logger.warning("Isolated atoms found in the structure")

        return {"atom": atoms, "bond": bonds, "state": state_attributes, "index1": index1, "index2": index2}

    @staticmethod
    def get_atom_features(structure) -> list:
        """
        Get atom features from structure, may be overwritten
        Args:
            structure: (Pymatgen.Structure) pymatgen structure
        Returns:
            list of atomic numbers
        """
        return np.array([i.specie.Z for i in structure], dtype="int32").tolist()

    def __call__(self, structure: Structure) -> dict:
        """
        Directly apply the converter to structure, alias to convert
        Args:
            structure (Structure): input structure

        Returns (dict): graph dictionary

        """
        return self.convert(structure)

    def get_input(self, structure: Structure) -> list[np.ndarray]:
        """
        Turns a structure into model input
        """
        graph = self.convert(structure)
        return self.graph_to_input(graph)

    def graph_to_input(self, graph: dict) -> list[np.ndarray]:
        """
        Turns a graph into model input
        Args:
            (dict): Dictionary description of the graph
        Return:
            ([np.ndarray]): Inputs in the form needed by MEGNet
        """
        gnode = [0] * len(graph["atom"])
        gbond = [0] * len(graph["index1"])

        return [
            expand_1st(self.atom_converter.convert(graph["atom"])),
            expand_1st(self.bond_converter.convert(graph["bond"])),
            expand_1st(np.array(graph["state"])),
            expand_1st(np.array(graph["index1"], dtype=np.int32)),
            expand_1st(np.array(graph["index2"], dtype=np.int32)),
            expand_1st(np.array(gnode, dtype=np.int32)),
            expand_1st(np.array(gbond, dtype=np.int32)),
        ]

    @staticmethod
    def get_flat_data(graphs: list[dict], targets: list | None = None) -> tuple:
        """
        Expand the graph dictionary to form a list of features and targets tensors.
        This is useful when the model is trained on assembled graphs on the fly.
        Args:
            graphs: (list of dictionary) list of graph dictionary for each structure
            targets: (list of float or list) Optional: corresponding target
                values for each structure
        Returns:
            tuple(node_features, edges_features, global_values, index1, index2, targets)
        """

        output = []  # Will be a list of arrays

        # Convert the graphs to matrices
        for feature in ["atom", "bond", "state", "index1", "index2"]:
            output.append([np.array(x[feature]) for x in graphs])

        # If needed, add the targets
        if targets is not None:
            output.append([to_list(t) for t in targets])

        return tuple(output)

    @staticmethod
    def _get_dummy_converter() -> DummyConverter:
        return DummyConverter()

    def as_dict(self) -> dict:
        """
        Serialize to dict
        Returns: (dict) dictionary of information
        """
        all_dict = super().as_dict()
        if "nn_strategy" in all_dict:
            nn_strategy = all_dict.pop("nn_strategy")
            all_dict.update({"nn_strategy": local_env.serialize(nn_strategy)})
        return all_dict

    @classmethod
    def from_dict(cls, d: dict) -> StructureGraph:
        """
        Initialization from dictionary
        Args:
            d (dict): dictionary

        Returns: StructureGraph object

        """
        if "nn_strategy" in d:
            nn_strategy = d.pop("nn_strategy")
            nn_strategy_obj = local_env.deserialize(nn_strategy)
            d.update({"nn_strategy": nn_strategy_obj})
            return super().from_dict(d)
        return super().from_dict(d)


class StructureGraphFixedRadius(StructureGraph):
    """
    This one uses a short cut to call find_points_in_spheres cython function in
    pymatgen. It is orders of magnitude faster than previous implementations
    """

    def convert(self, structure: Structure, state_attributes: list | None = None) -> dict:
        """
        Take a pymatgen structure and convert it to a index-type graph representation
        The graph will have node, distance, index1, index2, where node is a vector of Z number
        of atoms in the structure, index1 and index2 mark the atom indices forming the bond and separated by
        distance.
        For state attributes, you can set structure.state = [[xx, xx]] beforehand or the algorithm would
        take default [[0, 0]]
        Args:
            state_attributes: (list) state attributes
            structure: (pymatgen structure)
            (dictionary)
        """
        state_attributes = (
            state_attributes or getattr(structure, "state", None) or np.array([[0.0, 0.0]], dtype="float32")
        )
        atoms = self.get_atom_features(structure)
        index1, index2, _, bonds = get_graphs_within_cutoff(structure, self.nn_strategy.cutoff)

        if len(index1) == 0:
            raise RuntimeError("The cutoff is too small, resulting in " "material graph with no bonds")

        if np.size(np.unique(index1)) < len(atoms):
            logger.warning("Isolated atoms found in the structure. The " "cutoff radius might be small")

        return {"atom": atoms, "bond": bonds, "state": state_attributes, "index1": index1, "index2": index2}

    @classmethod
    def from_structure_graph(cls, structure_graph: StructureGraph) -> StructureGraphFixedRadius:
        """
        Initialize from pymatgen StructureGraph
        Args:
            structure_graph (StructureGraph): pymatgen StructureGraph object

        Returns: StructureGraphFixedRadius object

        """
        return cls(
            nn_strategy=structure_graph.nn_strategy,
            atom_converter=structure_graph.atom_converter,
            bond_converter=structure_graph.bond_converter,
        )


class DummyConverter(Converter):
    """
    Dummy converter as a placeholder
    """

    def convert(self, d):
        """
        Dummy convert, does nothing to input
        Args:
            d (Any): input object

        Returns: d

        """
        return d


class EmbeddingMap(Converter):
    """
    Convert an integer to a row vector in a feature matrix
    """

    def __init__(self, feature_matrix: np.ndarray):
        """
        Args:
            feature_matrix: (np.ndarray) A matrix of shape (N, M)
        """
        self.feature_matrix = np.array(feature_matrix)

    def convert(self, int_array: np.ndarray) -> np.ndarray:
        """
        convert atomic number to row vectors in the feature_matrix
        Args:
            int_array: (1d array) number array of length L
        Returns
            (matrix) L*M matrix with N the length of d and M the length of centers
        """
        return self.feature_matrix[int_array]


class GaussianDistance(Converter):
    """
    Expand distance with Gaussian basis sit at centers and with width 0.5.
    """

    def __init__(self, centers: np.ndarray = np.linspace(0, 5, 100), width=0.5):
        """

        Args:
            centers: (np.array) centers for the Gaussian basis
            width: (float) width of Gaussian basis
        """
        self.centers = centers
        self.width = width

    def convert(self, d: np.ndarray) -> np.ndarray:
        """
        expand distance vector d with given parameters
        Args:
            d: (1d array) distance array
        Returns
            (matrix) N*M matrix with N the length of d and M the length of centers
        """
        d = np.array(d)
        print(np.exp(-((d[:, None] - self.centers[None, :]) ** 2) / self.width**2))
        return np.exp(-((d[:, None] - self.centers[None, :]) ** 2) / self.width**2)


def itemgetter_list(data_list: list, indices: list) -> tuple:
    """
    Get indices of data_list and return a tuple
    Args:
        data_list (list):  data list
        indices: (list) indices
    Returns:
        (tuple)
    """
    it = itemgetter(*indices)
    if np.size(indices) == 1:
        return (it(data_list),)
    return it(data_list)
