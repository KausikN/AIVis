'''
Data Generators
'''

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from sklearn.datasets import make_blobs

# Main Vars
fig = plt.figure()
canvas = FigureCanvasAgg(fig)

# Main Functions
# Plot Functions
def PlotLabelledData(Dataset, title='', plot=True):
    '''
    Plots the data with labels.
    '''
    global fig, canvas
    fig = plt.figure()
    canvas = FigureCanvasAgg(fig)
    fig.clear()

    X = Dataset['points']
    labels = Dataset['labels']
    unique_labels = Dataset['unique_labels']
    centers = np.array([])
    if 'centers' in Dataset.keys():
        centers = Dataset['centers']

    # Init Plot
    ax = None
    # If 3D
    if Dataset['dim'] >= 3:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()

    # Plot the data
    for ul in unique_labels:
        X_ul = X[labels == ul]

        # If 3D
        if Dataset['dim'] >= 3:
            ax.scatter3D(X_ul[:, 0], X_ul[:, 1], X_ul[:, 2], label=ul)
        elif Dataset['dim'] == 2:
            ax.scatter(X_ul[:, 0], X_ul[:, 1], label=ul)
        elif Dataset['dim'] == 1:
            ax.scatter(X_ul[:, 0], np.zeros(X_ul.shape), label=ul)

    # Plot Centers
    if centers.shape[0] > 0:
        centersParams = {'marker': 'x', 'label': 'Centers', 's': 50, 'c': 'black'}
        if Dataset['dim'] >= 3:
            ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], **centersParams)
        elif Dataset['dim'] == 2:
            ax.scatter(centers[:, 0], centers[:, 1], **centersParams)
        elif Dataset['dim'] == 1:
            ax.scatter(centers[:, 0], np.zeros(centers.shape), **centersParams)

    # plt.legend()
    plt.title(title)

    canvas.draw()
    buf = canvas.buffer_rgba()
    I_plot = cv2.cvtColor(np.asarray(buf), cv2.COLOR_RGBA2RGB)
    
    # if plot:
    #     plt.show()
    plt.close(fig)

    return I_plot

# Generate Functions
def GenerateRandomBlobs(N, dim, centers, plot=False):
    '''
    Generates a random dataset of 2D points.
    '''
    # Init Dataset
    Dataset = {}
    # Generate random dataset
    X, y = make_blobs(n_samples=N, n_features=dim, centers=centers, random_state=42)
    Dataset['points'] = np.array(X)
    Dataset['labels'] = np.array(y)
    Dataset['unique_labels'] = np.unique(Dataset['labels'])
    Dataset['dim'] = dim

    # Plot the dataset
    if plot:
        PlotLabelledData(Dataset, title='Random Blobs')

    return Dataset

# Driver Code
# Params

# Params

# RunCode
# Dataset = GenerateRandomBlobs(N=200, dim=3, centers=None, plot=True)