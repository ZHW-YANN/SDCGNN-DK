import csv
import json
import os
import re

import pandas as pd
import torch
import numpy as np
from ray.tune.examples.cifar10_pytorch import load_data

from data_loader import data_utils, process_cif, GeoCGNN_data_utils
from torch_geometric.data import Data


def load_data(path):
    # Globally load atomic features
    dataSet = []
    get_atom_feather(os.path.join(path, 'atom_feature.csv'))
    all_data = process_cif.CIFData(path + '/cif')
    RT_targets = pd.read_csv('new_RT_targets.csv').values
    # RT_targets = pd.read_csv('RT_targets_test.csv').values
    for i in range(len(all_data)):
        one_data = []
        # Crystal structure Graph representation
        cs_x, cs_edge_attr, cs_edge_index, target, cif_id = all_data[i]
        # Interstice network Graph representation
        in_x, in_edge_attr, in_edge_index = get_interstice_network(path + '/txt', cif_id)

        one_data.append(cs_x)
        one_data.append(in_x)

        one_data.append(cs_edge_index)
        one_data.append(in_edge_index)

        one_data.append(cs_edge_attr)
        # one_data.append(in_edge_attr[:,0].reshape(-1, 1)) # mask weight matrix of energy
        one_data.append(in_edge_attr)

        one_data.append(torch.Tensor([RT_targets[i][1], RT_targets[i][2], RT_targets[i][3]]))

        # Ablation modification
        # one_data.append(torch.Tensor([0.0, 0.0, 0.0]))

        one_data.append(target)

        one_data.append(cif_id)

        dataSet.append(one_data)
        if len(dataSet) % 1000 == 0:
            print('The data is loading:', len(dataSet))
    print('All data has loaded completely!')
    return dataSet


def get_permutation_invariant_features(in_x, atoms_num):
    """
    Generate permutation invariant features for interstice networks

    Args:
        in_x: Input feature tensor
        atoms_num: Number of atoms

    Returns:
        Permutation invariant features
    """
    batch_size = in_x.shape[0]
    atom_feat_dim = len(atom_feature[0])

    # Extract atomic types and convert them into feature vectors
    atom_types = in_x[:, :atoms_num].long()  # Atomic type indices
    other_features = in_x[:, atoms_num:]  # Other features

    # Create feature vectors for each atom
    atom_embeddings = []
    for i in range(batch_size):
        sample_embeddings = []
        for atom_type in atom_types[i]:
            if atom_type != 0:  # Valid atom
                embedding = atom_feature[atom_type - 1]
            else:  # Padding or invalid position
                embedding = torch.zeros(atom_feat_dim)
            sample_embeddings.append(embedding)
        atom_embeddings.append(torch.stack(sample_embeddings))

    atom_embeddings = torch.stack(atom_embeddings)  # [batch, atoms_num, feat_dim]

    # Use permutation invariant operations: sum, mean, max, etc.
    aggregated_features = torch.sum(atom_embeddings, dim=1)  # Sum aggregation [batch, feat_dim]
    # Or use: torch.mean(atom_embeddings, dim=1)

    # Concatenate with other features
    invariant_features = torch.cat([aggregated_features, other_features], dim=1)

    return invariant_features


def get_interstice_network(path, id_file, atoms_num=12):
    """
    Get interstice network data including node features and edge information

    Args:
        path: Path to data files
        id_file: File identifier
        atoms_num: Number of atoms, default 12

    Returns:
        Node features, edge attributes, and edge indices for interstice network
    """
    interitice_file = id_file + '_adjacency_void_atoms2.txt'
    channel_file = id_file + '_adjacency_channel_atoms2.txt'
    in_x = get_x_data(os.path.join(path, interitice_file), atoms_num)
    in_edge_index, in_edge_attr = get_edge_data(os.path.join(path, channel_file), atoms_num)

    in_x = torch.tensor(in_x, dtype=torch.float)
    # Embed atomic features into interstices
    new_x = torch.zeros(in_x.shape[0], atoms_num, len(atom_feature[0])).float()
    for i in range(in_x.shape[0]):
        for j in range(atoms_num):
            if in_x[i][j] != 0:
                new_x[i][j] = atom_feature[in_x[i][j].int() - 1]
    new_x = new_x.view(in_x.shape[0], atoms_num * len(atom_feature[0]))
    in_x = get_permutation_invariant_features(in_x, atoms_num)
    in_edge_index = torch.tensor(in_edge_index, dtype=torch.long)
    in_edge_attr = torch.tensor(in_edge_attr, dtype=torch.float)

    return in_x, in_edge_attr, in_edge_index


def get_atom_feather(feature_path):
    """
    Load atomic features from file and preprocess

    Args:
        feature_path: Path to atomic feature CSV file
    """
    # Load atomic features
    feature = data_utils.load_csv(feature_path)
    feature = data_utils.min_max_normalization(feature)
    a = data_utils.toOneHot(feature[:, 9].reshape(len(feature), 1))
    b = data_utils.toOneHot(feature[:, 10].reshape(len(feature), 1))
    c = data_utils.toOneHot(feature[:, 13].reshape(len(feature), 1))
    feature = np.delete(feature, [9, 10, 13], axis=1)
    feature = np.c_[feature, a]
    feature = np.c_[feature, b]
    feature = np.c_[feature, c]
    global atom_feature

    atom_feature = torch.Tensor(feature)

def get_x_data(fileInfo, atoms_num):
    """
    Extract node feature data from file

    Args:
        fileInfo: Path to data file
        atoms_num: Number of atoms

    Returns:
        List of atom features
    """
    file = open(fileInfo, "r")
    atoms_feather = []
    for line in file.readlines():  # For each file
        line = line.strip('\n')  # Remove newline characters
        line = line.split('\t')  # Split string by tab delimiter
        feather_atoms = line[2]
        feather_atoms = feather_atoms.split(' ')
        if len(feather_atoms) > atoms_num:
            feather_atoms = feather_atoms[:atoms_num]
        while len(feather_atoms) < atoms_num:
            feather_atoms.append(0)
        feather_atoms = list(map(float, feather_atoms))
        feather_atoms_dis = line[3]
        feather_atoms_dis = feather_atoms_dis.split(' ')
        if len(feather_atoms_dis) > atoms_num:
            feather_atoms_dis = feather_atoms_dis[:atoms_num]
        while len(feather_atoms_dis) < atoms_num:
            feather_atoms_dis.append(0)
        feather_atoms_dis = list(map(float, feather_atoms_dis))

        feather_gap_radius = line[4]
        feather_gap_radius = feather_gap_radius.split(' ')
        feather_gap_radius = list(map(float, feather_gap_radius))

        feather_gap_energy = line[5]
        feather_gap_energy = feather_gap_energy.split(' ')
        feather_gap_energy = list(map(float, feather_gap_energy))
        if feather_gap_radius[0] < 0.0:
            print(fileInfo)
        feather = feather_atoms + feather_atoms_dis + feather_gap_radius + feather_gap_energy
        atoms_feather.append(feather)
    file.close()
    return atoms_feather


def get_edge_data(fileInfo, atoms_num):
    """
    Extract edge data from file

    Args:
        fileInfo: Path to data file
        atoms_num: Number of atoms

    Returns:
        Edge indices and edge attributes
    """
    file = open(fileInfo, "r")
    edge_index = []
    edge_attr = []
    edge_index_0 = []
    edge_index_1 = []
    for line in file.readlines():
        line = line.strip('\n')
        line = line.split('\t')
        feather = []
        for i in range(len(line)):
            if i == 0:
                edge_index_item = re.findall('[0-9]+', line[i])
                edge_index_0.append(edge_index_item[0])
                edge_index_1.append(edge_index_item[1])
            if i == 1 or i == 2:
                feather_atoms_or_dis = line[i]
                feather_atoms_or_dis = feather_atoms_or_dis.split(' ')
                if len(feather_atoms_or_dis) > atoms_num:
                    feather_atoms_or_dis = feather_atoms_or_dis[:atoms_num]
                while len(feather_atoms_or_dis) < atoms_num:
                    feather_atoms_or_dis.append(0)
                feather_atoms_or_dis = list(map(float, feather_atoms_or_dis))
                feather += feather_atoms_or_dis
            if i == 4:
                feather_bottle_radius = line[i]
                feather_bottle_radius = feather_bottle_radius.split(' ')
                feather_bottle_radius = list(map(float, feather_bottle_radius))
                feather += feather_bottle_radius
            if i == 6:
                feather_energy_difference = line[i]
                feather_energy_difference = feather_energy_difference.split(' ')
                feather_energy_difference = list(map(float, feather_energy_difference))
                feather += feather_energy_difference
        edge_attr.append(feather)
    edge_index.append(list(map(int, edge_index_0)))
    edge_index.append(list(map(int, edge_index_1)))
    file.close()
    return edge_index, edge_attr


if __name__ == '__main__':
    dataset = load_data('../data')
