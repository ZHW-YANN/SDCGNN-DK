import numpy as np
import torch
import csv
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


# Convert features to one-hot encoding
def toOneHot(data):
    """
    Convert categorical data to one-hot encoded format

    Args:
        data: Input categorical data

    Returns:
        One-hot encoded array representation of input data
    """
    enc = OneHotEncoder()
    enc.fit(data)
    result = enc.transform(data).toarray()
    return result


def load_csv(path):
    """
    Load data from CSV file and convert to numpy array

    Args:
        path: Path to the CSV file

    Returns:
        Numpy array containing the loaded data
    """
    file = open(path, "r")
    reader = csv.reader(file)
    result = []
    for item in reader:
        # Skip header row
        if reader.line_num == 1:
            continue
        # Extract data excluding first column and convert to float
        temp = item[1:len(item)]
        temp = list(map(float, temp))
        result.append(temp)
    result = np.array(result)
    return result


def min_max_normalization(data):
    """
    Apply min-max normalization to the input data

    Args:
        data: Input data to be normalized

    Returns:
        Normalized data with values scaled between 0 and 1
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    result = min_max_scaler.fit_transform(data)
    return result


def load_dataSet(path):
    """
    Load dataset from file and process into graph adjacency matrices

    Args:
        path: Path to the dataset file

    Returns:
        List of data samples, each containing adjacency matrix and label
    """
    file = open(path, "r")
    dataSet = []
    for line in file.readlines():
        data = []
        line = line.strip('\n')
        line = line.split('\t')
        # Remove first and last elements
        del line[-1]
        del line[0]
        row = []
        # Process adjacency matrix data
        for adj_row_num in range(1, len(line)):
            temp = line[adj_row_num]
            temp = temp.split(' ')
            row.append(list(map(float, temp)))
        row = np.array(row)
        row = min_max_normalization(row)  # Normalize adjacency matrix by columns
        data.append(torch.tensor(row, dtype=torch.float))
        data.append(torch.tensor(float(line[0]), dtype=torch.float))
        dataSet.append(data)
    return dataSet
