import json
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from scipy.special import jn, sph_harm, jn_zeros


def process(config, data_path, radius=8, max_num_nbr=12):
    """
    Process crystal structure data and extract atomic features, neighbor information 
    and various structural feature vectors

    Args:
        config (dict): Configuration dictionary containing atomic numbers and node vectors
        data_path (str): Crystal structure file path (e.g. CIF file)
        radius (float): Neighbor search radius, default 8Å
        max_num_nbr (int): Maximum number of neighbors, default 12

    Returns:
        nodes (np.ndarray): Atomic feature matrix with shape (n_atoms, n_features)
        edge_distance (np.ndarray): Edge distance features with shape (n_edges,)
        edge_sources (np.ndarray): Source node indices of edges with shape (n_edges,)
        edge_targets (np.ndarray): Target node indices of edges with shape (n_edges,)
        combine_sets (np.ndarray): Gaussian radial basis combination features with shape (n_edges, 64)
        plane_wave (np.ndarray): Plane wave features with shape (n_edges, grid^3)
    """

    crystal = Structure.from_file(data_path)
    volume = crystal.lattice.volume
    coords = crystal.cart_coords
    lattice = crystal.lattice.matrix
    atoms = crystal.atomic_numbers
    material_id = data_path[:-4]
    atomnum = config['atomic_numbers']
    z_dict = {z: i for i, z in enumerate(atomnum)}
    one_hotvec = np.array(config["node_vectors"])
    atom_fea = np.vstack([one_hotvec[z_dict[atoms[i]]] for i in range(len(crystal))])

    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []

    for i, nbr in enumerate(all_nbrs):
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append(list(map(lambda x: x[2].tolist(), nbr)) +
                               [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[0].coords.tolist(), nbr)) +
                           [[coords[i][0] + radius, coords[i][1], coords[i][2]]] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2].tolist(),
                                        nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[0].coords.tolist(),
                                    nbr[:max_num_nbr])))
    atom_fea = atom_fea.tolist()

    nbr_subtract = []
    nbr_distance = []

    for i in range(len(nbr_fea)):
        if nbr_fea[i] != []:
            x = nbr_fea[i] - coords[:, np.newaxis, :][i]
            nbr_subtract.append(x)
            nbr_distance.append(np.linalg.norm(x, axis=1).tolist())
        else:
            nbr_subtract.append(np.array([]))
            nbr_distance.append(np.array([]))

    nbr_fea_idx = np.array(nbr_fea_idx)

    nei = nbr_fea_idx
    distance = nbr_distance
    vector = nbr_subtract
    n_nodes = len(atom_fea)
    nodes = np.array(atom_fea, dtype=np.float32)
    edge_sources = np.concatenate([[i] * len(nei[i]) for i in range(n_nodes)])
    edge_targets = np.concatenate(nei)
    edge_vector = np.array(vector, dtype=np.float32)
    edge_index = np.concatenate([range(len(nei[i])) for i in range(n_nodes)])
    vectorij = edge_vector[edge_sources, edge_index]
    edge_distance = np.array(distance, dtype=np.float32)
    distance = edge_distance[edge_sources, edge_index]
    combine_sets = []
    # gaussian radial
    N = 64
    cutoff = 8
    for n in range(1, N + 1):
        phi = Phi(distance, cutoff)
        G = gaussian(distance, miuk(n, N, cutoff), betak(N, cutoff))
        combine_sets.append(phi * G)
    combine_sets = np.array(combine_sets, dtype=np.float32).transpose()

    # plane wave
    grid = 4
    kr = np.dot(vectorij, get_Kpoints_random(grid, lattice, volume).transpose())
    plane_wave = np.cos(kr) / np.sqrt(volume)
    return nodes, edge_distance, edge_sources, edge_targets, combine_sets, plane_wave


def a_SBF(alpha, l, n, d, cutoff):
    """
    Calculate spherical Bessel function features

    Args:
        alpha: Angular coordinate
        l: Order of spherical harmonic
        n: Order of Bessel function
        d: Distance parameter
        cutoff: Cutoff radius

    Returns:
        Spherical Bessel function value
    """
    root = float(jn_zeros(l, n)[n - 1])
    return jn(l, root * d / cutoff) * sph_harm(0, l, np.array(alpha), 0).real * np.sqrt(
        2 / cutoff ** 3 / jn(l + 1, root) ** 2)


def a_RBF(n, d, cutoff):
    """
    Calculate radial basis function

    Args:
        n: Basis function index
        d: Distance parameter
        cutoff: Cutoff radius

    Returns:
        Radial basis function value
    """
    return np.sqrt(2 / cutoff) * np.sin(n * np.pi * d / cutoff) / d


def get_Kpoints_random(q, lattice, volume):
    """
    Generate random K-points for plane wave calculations

    Args:
        q: Grid size parameter
        lattice: Lattice matrix
        volume: Unit cell volume

    Returns:
        Array of K-points
    """
    a0 = lattice[0, :]
    a1 = lattice[1, :]
    a2 = lattice[2, :]
    unit = 2 * np.pi * np.vstack((np.cross(a1, a2), np.cross(a2, a0), np.cross(a0, a1))) / volume
    ur = [(2 * r - q - 1) / 2 / q for r in range(1, q + 1)]
    points = []
    for i in ur:
        for j in ur:
            for k in ur:
                points.append(unit[0, :] * i + unit[1, :] * j + unit[2, :] * k)
    points = np.array(points)
    return points


def Phi(r, cutoff):
    """
    Calculate cutoff function to ensure smooth decay at boundary

    Args:
        r: Distance
        cutoff: Cutoff radius

    Returns:
        Cutoff function value
    """
    return 1 - 6 * (r / cutoff) ** 5 + 15 * (r / cutoff) ** 4 - 10 * (r / cutoff) ** 3


def gaussian(r, miuk, betak):
    """
    Calculate Gaussian-type radial function

    Args:
        r: Distance
        miuk: Mean parameter
        betak: Width parameter

    Returns:
        Gaussian function value
    """
    return np.exp(-betak * (np.exp(-r) - miuk) ** 2)


def miuk(n, K, cutoff):
    """
    Calculate mean parameter for Gaussian functions

    Args:
        n: Function index
        K: Total number of functions
        cutoff: Cutoff radius

    Returns:
        Mean parameter value
    """
    # n=[1,K]
    return np.exp(-cutoff) + (1 - np.exp(-cutoff)) / K * n


def betak(K, cutoff):
    """
    Calculate width parameter for Gaussian functions

    Args:
        K: Total number of functions
        cutoff: Cutoff radius

    Returns:
        Width parameter value
    """
    return (2 / K * (1 - np.exp(-cutoff))) ** (-2)
