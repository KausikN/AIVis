'''
Activation Functions
'''

# Imports
import numpy as np

# Main Functions
# Activation Functions
def sigmoid(x):
    '''
    Sigmoid Function
    '''
    return 1 / (1 + np.exp(-x))

def tanh(x):
    '''
    Tanh Function
    '''
    return np.tanh(x)

def relu(x):
    '''
    ReLU Function
    '''
    return np.maximum(0, x)

def leaky_relu(x):
    '''
    Leaky ReLU Function
    '''
    return np.maximum(0.01 * x, x)

def softmax(x):
    '''
    Softmax Function
    '''
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def identity(x):
    '''
    Identity Function
    '''
    return x

def softplus(x):
    '''
    Softplus Function
    '''
    return np.log(1 + np.exp(x))

def softsign(x):
    '''
    Softsign Function
    '''
    return x / (1 + np.abs(x))

def exponential(x):
    '''
    Exponential Function
    '''
    return np.exp(x)

def linear(x):
    '''
    Linear Derivative Function
    '''
    return 1

# Derivatives Functions
def sigmoid_deriv(x):
    '''
    Sigmoid Derivative Function
    '''
    return x * (1 - x)

def tanh_deriv(x):
    '''
    Tanh Derivative Function
    '''
    return 1 - (x ** 2)

def relu_deriv(x):
    '''
    ReLU Derivative Function
    '''
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def leaky_relu_deriv(x):
    '''
    Leaky ReLU Derivative Function
    '''
    x[x <= 0] = 0.01
    x[x > 0] = 1
    return x

def softmax_deriv(x):
    '''
    Softmax Derivative Function
    '''
    return x * (1 - x)

def identity_deriv(x):
    '''
    Identity Derivative Function
    '''
    return 1

def softplus_deriv(x):
    '''
    Softplus Derivative Function
    '''
    return 1 / (1 + np.exp(-x))

def softsign_deriv(x):
    '''
    Softsign Derivative Function
    '''
    return 1 / (1 + np.abs(x))

def exponential_deriv(x):
    '''
    Exponential Derivative Function
    '''
    return np.exp(x)

def linear_deriv(x):
    '''
    Linear Derivative Function
    '''
    return 1

# Driver Code
# Params

# Params

# RunCode