"""
Visualiser for datasets using python
"""

# Imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Main Vars
INPUT_DATA_TYPES = {
    "int": [int, np.int64], 
    "float": [float, np.float64], 
    "text": [str, np.object]
}

# Main Functions
# Basic Dataset Functions
def LoadDataset(path):
    '''
    Loads dataset from path
    '''
    return pd.read_csv(path)

def SaveDataset(dataset, path):
    '''
    Saves dataset to path
    '''
    dataset.to_csv(path, index=False)

# Data types Functions
def GetDatasetTypes(data):
    '''
    Returns the type of each data column
    '''

    types_data = data.dtypes
    types = []
    
    for t in types_data:
        this_type = "unknown"
        for k in INPUT_DATA_TYPES.keys():
            if t in INPUT_DATA_TYPES[k]:
                this_type = k
                break
        types.append(this_type)
    
    return types

def IsCategorizable(data, col, categoryThreshold=[10, 0.1]):
    '''
    Returns if a column is categorizable
    '''
    unique_values_count = len(GetUniqueValues(data, col))
    values_count = len(data[col])
    if unique_values_count <= categoryThreshold[0]:
        return True
    if (unique_values_count / values_count) <= categoryThreshold[1]:
        return True

# Data Functions
def GetUniqueValues(data, col):
    '''
    Returns the unique values in a column
    '''
    return data[col].unique()

def GetUniqueValuesCounts(data, col):
    '''
    Returns the unique values and their counts in a column
    '''
    return data[col].value_counts()

def GetMean(data, col):
    '''
    Returns the mean of a column
    '''
    return data[col].mean()

def GetMedian(data, col):
    '''
    Returns the median of a column
    '''
    return data[col].median()

def GetMode(data, col):
    '''
    Returns the mode of a column
    '''
    return data[col].mode()

# Driver Code
# # Params
# datasetPath = "Data/TestData/Finance.csv"
# # Params

# # RunCode
# dataset = LoadDataset(datasetPath)
# print(GetMean(dataset, "Year"))
# print(GetMedian(dataset, "Year"))
# print(GetMode(dataset, "Year"))
# print(GetUniqueValues(dataset, "Year"))
# print(GetUniqueValuesCounts(dataset, "Year"))