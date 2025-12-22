import numpy as np
import torch
import csv
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from typing import Any, Callable, List, Optional, Union, Tuple
from pymatgen.core import Molecule, Structure
from pymatgen.optimization.neighbors import find_points_in_spheres
from collections.abc import Iterable
StructureOrMolecule = Union[Structure, Molecule]

DTYPES = {
    "float32": {"numpy": np.float32},
    "float16": {"numpy": np.float16},
    "int32": {"numpy": np.int32},
    "int16": {"numpy": np.int16},
}


class DataType:
    """
    Data types for tensorflow. This enables users to choose
    from 32-bit float and int, and 16-bit float and int
    """

    np_float = np.float32
    np_int = np.int32

    @classmethod
    def set_dtype(cls, data_type: str) -> None:
        """
        Class method to set the data types
        Args:
            data_type (str): '16' or '32'
        """
        if data_type.endswith("32"):
            float_key = "float32"
            int_key = "int32"
        elif data_type.endswith("16"):
            float_key = "float16"
            int_key = "int16"
        else:
            raise ValueError("Data type not known, choose '16' or '32'")

        cls.np_float = DTYPES[float_key]["numpy"]
        cls.np_int = DTYPES[int_key]["numpy"]


def get_graphs_within_cutoff(
    structure: StructureOrMolecule, cutoff: float = 5.0, numerical_tol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get graph representations from structure within cutoff
    Args:
        structure (pymatgen Structure or molecule)
        cutoff (float): cutoff radius
        numerical_tol (float): numerical tolerance

    Returns:
        center_indices, neighbor_indices, images, distances
    """
    if isinstance(structure, Structure):
        lattice_matrix = np.ascontiguousarray(np.array(structure.lattice.matrix), dtype=float)
        pbc = np.array([1, 1, 1], dtype=int)
    elif isinstance(structure, Molecule):
        lattice_matrix = np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]], dtype=float)
        pbc = np.array([0, 0, 0], dtype=int)
    else:
        raise ValueError("structure type not supported")
    r = float(cutoff)
    cart_coords = np.ascontiguousarray(np.array(structure.cart_coords), dtype=float)
    center_indices, neighbor_indices, images, distances = find_points_in_spheres(
        cart_coords, cart_coords, r=r, pbc=pbc, lattice=lattice_matrix, tol=numerical_tol
    )
    center_indices = center_indices.astype(DataType.np_int)
    neighbor_indices = neighbor_indices.astype(DataType.np_int)
    images = images.astype(DataType.np_int)
    distances = distances.astype(DataType.np_float)
    exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)
    return center_indices[exclude_self], neighbor_indices[exclude_self], images[exclude_self], distances[exclude_self]


def to_list(x: Union[Iterable, np.ndarray]) -> List:
    """
    If x is not a list, convert it to list
    """
    if isinstance(x, Iterable):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()  # noqa
    return [x]


def expand_1st(x: np.ndarray) -> np.ndarray:
    """
    Adding an extra first dimension

    Args:
        x: (np.array)
    Returns:
         (np.array)
    """
    return np.expand_dims(x, axis=0)
