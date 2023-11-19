"""
Simple Fully Connected Neural Network
"""

# Imports
import json
import functools
import numpy as np
from numpy.lib.npyio import save
from tqdm import tqdm

from .Utils import VideoUtils
from .Utils import NetworkVis
from .FunctionsLibrary import LossFunctions
from .FunctionsLibrary import ActivationFunctions

# Utils Functions
def GenerateHistoryVideo(history, savePath, duration=2.0):
    Is = []
    print("Generating", history["n_iters"], "frames...")
    for i in tqdm(range(history["n_iters"])):
        nodes_vals = []
        for j in range(len(history["nodes"][i])):
            nodes_vals.extend(history["nodes"][i][j])
        nodes_range = [min(nodes_vals), max(nodes_vals)]

        weights_vals = []
        for j in range(len(history["Ws"][i])):
            flat_ws = np.array(history["Ws"][i][j]).flatten()
            weights_vals.extend(flat_ws)
        weights_range = [min(weights_vals), max(weights_vals)]

        network = {
            "nodes": history["nodes"][i],
            "weights": history["Ws"][i],
            "node_range": nodes_range,
            "weight_range": weights_range
        }
        I = NetworkVis.GenerateNetworkImage(network)
        # Is.append(np.array(I, dtype=np.uint8))
        Is.append(np.zeros((I.shape[0], I.shape[1], 3), dtype=np.uint8))

    fps = (len(Is) / duration)
    print("FPS: ", fps)
    VideoUtils.SaveFrames2Video(Is, savePath, fps)

# Main Functions
# Init Functions
def initialize_parameters(layer_sizes, funcs):
    Ws = []
    bs = []
    for i in range(len(layer_sizes)-1):
        W = np.random.randn(layer_sizes[i+1], layer_sizes[i])
        b = 0
        Ws.append(W)
        bs.append(b)

    parameters = {
        "n_layers": len(layer_sizes),
        "layer_sizes": layer_sizes,
        "Ws": Ws,
        "bs" : bs,
        "act_fn": {
            "func": funcs["act_fn"],
            "deriv": funcs["act_fn_deriv"]
        },
        "loss_fn": {
            "func": funcs["loss_fn"],
            "deriv": funcs["loss_fn_deriv"]
        }
    }
    return parameters

# Forward Propagation Functions
def forward_prop(X, parameters):
    As = [list(X.flatten())]

    Ws = parameters["Ws"]
    bs = parameters["bs"]
    act_fn = parameters["act_fn"]["func"]
    # Initial a = x
    a = X
    for i in range(len(Ws)):
        # o = W a(previous layer) + b
        Wa = np.dot(a, Ws[i].T)
        o = Wa + bs[i]
        # a = activation(o)
        a = act_fn(o)
        # Save all activations
        As.append(list(o.flatten()))

    return a, As

# Backward Propogation Functions
def backward_prop(X, y, parameters):
    n_layers = parameters["n_layers"]
    Ws = parameters["Ws"]
    bs = parameters["bs"]
    act_fn = parameters["act_fn"]["func"]
    act_fn_deriv = parameters["act_fn"]["deriv"]
    loss_fn_deriv = parameters["loss_fn"]["deriv"]

    grads = {}
    
    a = X
    node_values = [np.copy(a)]
    # Find final activations
    for i in range(len(Ws)):
        o = np.dot(a, Ws[i].T) + bs[i]
        a = act_fn(o)
        node_values.append(np.copy(a))

    # Find loss Derivative at output layer
    grads["dE"] = loss_fn_deriv(a, y)
    grads["dA"] = act_fn_deriv(a)
    grads["dO"] = []
    for i in range(n_layers):
        grads["dO"].append([])
    grads["dO"][-1] = grads["dE"] * grads["dA"]
    grads["dW"] = []
    grads["db"] = []
    # Find grads of nodes by going from last layer to first layer
    layer_indices = list(reversed(range(n_layers-1)))
    for i in layer_indices:
        dO = grads["dO"][i+1]
        # Find dW
        dW = np.dot(dO.T, node_values[i])
        grads["dW"].insert(0, dW)
        # Find db
        db = np.sum(bs[i] * dO)
        grads["db"].insert(0, db)
        # Find dO for ith layer
        dO_i = np.dot(dO, Ws[i])
        grads["dO"][i] = act_fn_deriv(node_values[i]) * dO_i

    return grads

def update_parameters(parameters, grads, lr):
    Ws = parameters["Ws"]
    bs = parameters["bs"]
    for i in range(len(Ws)):
        Ws[i] -= lr * grads["dW"][i]
        bs[i] -= lr * grads["db"][i]
    parameters["Ws"] = Ws
    parameters["bs"] = bs

    return parameters

# Model Functions
def model(X, Y, layer_sizes, n_epochs, lr, funcs):
    history = {
        "n_iters": n_epochs * X.shape[0],
        "layer_sizes": layer_sizes,
        "nodes": [],
        "Ws": [],
        "bs": [],
        "loss": []
    }

    parameters = initialize_parameters(layer_sizes, funcs)
    for i in tqdm(range(0, n_epochs)):
        for x, y in zip(X, Y):
            x = x.reshape(1, x.shape[0])
            y = y.reshape(1, y.shape[0])

            y_out, As = forward_prop(x, parameters)
            loss = funcs["loss_fn"](y_out, y)
            grads = backward_prop(x, y, parameters)
            parameters = update_parameters(parameters, grads, lr)
            
            # Record History
            # Convert Ws and bs to lists
            bs = parameters["bs"]
            Ws = []
            for W in parameters["Ws"]:
                W = (W.T).tolist()
                Ws.append(W)

            bs = list(bs)
            history["nodes"].append(As)
            history["Ws"].append(Ws)
            history["bs"].append(bs)
            history["loss"].append(loss)

        if(i%1 == 0):
            print(f"EPOCH {i}: {loss}")

    return parameters, history

# Predict Functions
def predict(X, parameters):
    y_pred, _ = forward_prop(X, parameters)
    y_pred = np.squeeze(y_pred)
    return y_pred >= 0.5

# Driver Code
# Params
network_layers = [4]

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
Y = [[0, 0], [1, 1], [1, 1], [0, 0]]

learning_rate = 0.3
n_epochs = 5
funcs = {
    "act_fn": ActivationFunctions.sigmoid,
    "act_fn_deriv": ActivationFunctions.sigmoid_deriv,
    "loss_fn": LossFunctions.categorical_cross_entropy_error,
    "loss_fn_deriv": LossFunctions.categorical_cross_entropy_error_deriv
}

savePath = "GeneratedVisualisations/Haha.avi"
duration = 1.0
# Params

# RunCode
X = np.array(X)
Y = np.array(Y)
network_layers = [X.shape[1]] + network_layers + [Y.shape[1]]

# Train Model
trained_parameters, history = model(X, Y, network_layers, n_epochs, learning_rate, funcs=funcs)

# Generate Video
tempSavePath = "GeneratedVisualisations/temp.avi"
GenerateHistoryVideo(history, tempSavePath, duration)

# Fix Video
VideoUtils.FixVideoFile(tempSavePath, savePath)