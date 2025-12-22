import csv
import json
import os
import re

import pandas as pd
import torch
import numpy as np

from data_loader import data_utils, process_cif, GeoCGNN_data_utils
from torch_geometric.data import Data


# MEGNET
def load_data_MEGNET(path):
    # 全局加载原子特征
    dataSet = []
    all_data = process_cif.CIFData(path + '/cif')
    RT_targets = pd.read_csv('new_RT_targets_min_max.csv').values
    for i in range(len(all_data)):
        one_data = []
        # CGCNN的图表征
        cs_x, cs_edge_attr, cs_edge_index, target, cif_id = all_data[i]
        one_data.append(cs_x)
        one_data.append(cs_edge_index)
        one_data.append(cs_edge_attr)
        one_data.append(torch.Tensor([0.0, 0.0]))
        one_data.append(target)
        one_data.append(cif_id)
        dataSet.append(one_data)
        if len(dataSet) % 1000 == 0:
            print('The data is loading：', len(dataSet))
    print('All data has loaded completely！Totally having：', len(dataSet))
    return dataSet




