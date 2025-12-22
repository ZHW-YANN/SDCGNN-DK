import numpy as np
import torch
import csv
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

#特征转为one-hot
def toOneHot(data):
    enc = OneHotEncoder()
    enc.fit(data)
    result = enc.transform(data).toarray()
    return result

def load_csv(path):
    file = open(path, "r")
    reader = csv.reader(file)
    result = []
    for item in reader:
        if reader.line_num == 1:
            continue
        temp = item[1:len(item)]
        temp = list(map(float, temp))
        result.append(temp)
    result = np.array(result)
    return result


def min_max_normalization(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    result = min_max_scaler.fit_transform(data)
    return result


def load_dataSet(path):
    file = open(path, "r")
    dataSet = []
    for line in file.readlines():
        data = []
        line = line.strip('\n')
        line = line.split('\t')
        del line[-1]
        del line[0]
        row = []
        for adj_row_num in range(1, len(line)):
            temp = line[adj_row_num]
            temp = temp.split(' ')
            row.append(list(map(float, temp)))
        row = np.array(row)
        row = min_max_normalization(row)  # row是列归一化后的邻接矩阵
        data.append(torch.tensor(row, dtype=torch.float))
        data.append(torch.tensor(float(line[0]), dtype=torch.float))
        dataSet.append(data)
    return dataSet


