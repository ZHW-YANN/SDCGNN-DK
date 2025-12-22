import csv
import json
import os
import re

import pandas as pd
import torch
import numpy as np

from data_loader import data_utils, process_cif, GeoCGNN_data_utils
from torch_geometric.data import Data


def load_data2(path, config):
    # 全局加载原子特征
    dataSet = []
    get_atom_feather(os.path.join(path, 'atom_feature.csv'))
    data_path = os.path.join(path+'/cif', 'id_prop.csv')
    all_data = pd.read_csv(data_path).values
    for i in range(len(all_data)):
        one_data = []
        # GeoCGNN的图表征
        cif_id = all_data[i][0]
        cif_path = os.path.join(path+'/cif', all_data[i][0]+'.cif')
        cs_x, cs_edge_distance, cs_edge_sources, cs_edge_targets, cs_combine_sets, cs_plane_wave = GeoCGNN_data_utils.process(config, cif_path)
        target = all_data[i][1]
        # GeoCGNN的图表征
        # 间隙网络的图表征
        in_x, in_edge_attr, in_edge_index = get_interitice_network(path + '/txt', all_data[i][0])
        one_data.append(cs_x)
        one_data.append(cs_edge_distance)
        one_data.append(cs_edge_sources)
        one_data.append(cs_edge_targets)
        one_data.append(cs_combine_sets)
        one_data.append(cs_plane_wave)

        one_data.append(in_x)
        one_data.append(in_edge_attr)
        one_data.append(in_edge_index)

        one_data.append(target)

        one_data.append(cif_id)
        dataSet.append(one_data)
        if len(dataSet) % 100 == 0:
            print('The data is loading：', len(dataSet))
    print('All data has loaded completely！Totally having：', len(dataSet))
    return dataSet
# CGCNN
def load_data_CGCNN(path):
    # 全局加载原子特征
    dataSet = []
    all_data = process_cif.CIFData(path + '/cif')
    for i in range(len(all_data)):
        one_data = []
        # CGCNN的图表征
        cs_x, cs_edge_attr, cs_edge_index, target, cif_id = all_data[i]
        one_data.append(cs_x)
        one_data.append(cs_edge_index)
        one_data.append(cs_edge_attr)
        one_data.append(target)
        one_data.append(cif_id)
        dataSet.append(one_data)
        if len(dataSet) % 1000 == 0:
            print('The data is loading：', len(dataSet))
    print('All data has loaded completely！Totally having：', len(dataSet))
    return dataSet
def get_interitice_network(path, id_file, atoms_num=12):
    interitice_file = id_file + '_adjacency_void_atoms2.txt'
    channel_file = id_file + '_adjacency_channel_atoms2.txt'
    in_x = get_x_data(os.path.join(path, interitice_file), atoms_num)
    in_edge_index, in_edge_attr = get_edge_data(os.path.join(path, channel_file), atoms_num)

    in_x = torch.tensor(in_x, dtype=torch.float)
    # 原子特征嵌入间隙
    new_x = torch.zeros(in_x.shape[0], atoms_num, len(atom_feature[0])).float()
    for i in range(in_x.shape[0]):
        for j in range(atoms_num):
            if in_x[i][j] != 0:
                new_x[i][j] = atom_feature[in_x[i][j].int() - 1]
    new_x = new_x.view(in_x.shape[0], atoms_num * len(atom_feature[0]))
    in_x = torch.cat((new_x, in_x[:, atoms_num:]), dim=1)
    # 原子特征嵌入瓶颈
    in_edge_index = torch.tensor(in_edge_index, dtype=torch.long)
    in_edge_attr = torch.tensor(in_edge_attr, dtype=torch.float)
    new_edge_attr = torch.zeros(in_edge_attr.shape[0], 12, len(atom_feature[0])).float()
    for i in range(in_edge_attr.shape[0]):
        for j in range(12):
            if in_edge_attr[i][j] != 0:
                new_edge_attr[i][j] = atom_feature[in_edge_attr[i][j].int() - 1]
    new_edge_attr = new_edge_attr.view(in_edge_attr.shape[0], 12 * len(atom_feature[0]))
    in_edge_attr = torch.cat((new_edge_attr, in_edge_attr[:, 12:]), dim=1)

    return in_x, in_edge_attr, in_edge_index


def get_atom_feather(feature_path):
    # 加载原子特征
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
    file = open(fileInfo, "r")
    atoms_feather = []
    for line in file.readlines():  # 对于每个文件
        line = line.strip('\n')  # 去掉换行符
        line = line.split('\t')  # Python split()通过指定分隔符对字符串进行切片，此处是以tab分割字符串line。
        feather_atoms = line[1]
        feather_atoms = feather_atoms.split(' ')
        if len(feather_atoms) > atoms_num:
            feather_atoms = feather_atoms[:atoms_num]
        while len(feather_atoms) < atoms_num:
            feather_atoms.append(0)
        feather_atoms = list(map(float, feather_atoms))
        feather_atoms_dis = line[2]
        feather_atoms_dis = feather_atoms_dis.split(' ')
        if len(feather_atoms_dis) > atoms_num:
            feather_atoms_dis = feather_atoms_dis[:atoms_num]
        while len(feather_atoms_dis) < atoms_num:
            feather_atoms_dis.append(0)
        feather_atoms_dis = list(map(float, feather_atoms_dis))
        feather_gap_radius = line[3]
        feather_gap_radius = feather_gap_radius.split(' ')
        feather_gap_radius = list(map(float, feather_gap_radius))
        feather = feather_atoms + feather_atoms_dis + feather_gap_radius
        atoms_feather.append(feather)
    file.close()
    return atoms_feather


def get_edge_data(fileInfo, atoms_num):
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
            if i == 3:
                feather_gap_radius = line[i]
                feather_gap_radius = feather_gap_radius.split(' ')
                feather_gap_radius = list(map(float, feather_gap_radius))
                feather += feather_gap_radius
        edge_attr.append(feather)
    edge_index.append(list(map(int, edge_index_0)))
    edge_index.append(list(map(int, edge_index_1)))
    file.close()
    return edge_index, edge_attr