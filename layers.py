import numpy as np
from activations import relu, sigmoid, relu_backward, sigmoid_backward
from typing import Tuple, Any

def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray):
    """
    Implement the linear part of a layer's forward propagation.

    Parameters:
        A (np.ndarray): activations from previous layer (or input data): (size of previous layer, number of examples)
        W (np.ndarray): weights matrix: (size of current layer, size of previous layer)
        b (np.ndarray): bias vector, (size of the current layer, 1)

    Returns:
        Z (np.ndarray): the input of the activation function, also called pre-activation parameter
        cache (tuple): a tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_backward(dZ: np.ndarray, cache: Tuple[np.ndarray, np.ndarray, np.ndarray]):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Parameters:
        dZ (np.ndarray): Gradient of the cost with respect to the linear output (of current layer l)
        cache (tuple): tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
        dA_prev (np.ndarray): Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW (np.ndarray): Gradient of the cost with respect to W (current layer l), same shape as W
        db (np.ndarray): Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Parameters:
        A_prev (np.ndarray): activations from previous layer (or input data): (size of previous layer, number of examples)
        W (np.ndarray): weights matrix: (size of current layer, size of previous layer)
        b (np.ndarray): bias vector, (size of the current layer, 1)
        activation (str): the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
        A (np.ndarray): the output of the activation function, also called the post-activation value
        cache (tuple): a tuple containing "linear_cache" and "activation_cache";
                       stored for computing the backward pass efficiently
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def linear_activation_backward(dA: np.ndarray, cache: Tuple[Any, np.ndarray], activation: str):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Parameters:
        dA (np.ndarray): post-activation gradient for current layer
        cache (tuple): contains the linear_cache and the activation_cache
        activation (str): the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
        dA_prev (np.ndarray): Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW (np.ndarray): Gradient of the cost with respect to W (current layer l), same shape as W
        db (np.ndarray): Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db