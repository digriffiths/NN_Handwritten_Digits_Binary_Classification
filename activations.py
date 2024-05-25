import numpy as np

def sigmoid(Z: np.ndarray) -> tuple:
    """
    Compute the sigmoid activation of the input.

    Args:
    Z (np.ndarray): The input numpy array for which to compute the sigmoid activation.

    Returns:
    tuple: A tuple containing the sigmoid activation (A) and the input (Z) as the activation cache.
    """
    A = 1 / (1 + np.exp(-Z))
    activation_cache = Z
    return A, activation_cache

def relu(Z: np.ndarray) -> tuple:
    """
    Compute the ReLU activation of the input.

    Args:
    Z (np.ndarray): The input numpy array for which to compute the ReLU activation.

    Returns:
    tuple: A tuple containing the ReLU activation (A) and the input (Z) as the activation cache.
    """
    A = np.maximum(0, Z)
    activation_cache = Z
    return A, activation_cache

def relu_backward(dA: np.ndarray, activation_cache: np.ndarray) -> np.ndarray:
    """
    Compute the backward propagation for a single ReLU unit.

    Args:
    dA (np.ndarray): The post-activation gradient.
    activation_cache (np.ndarray): The activation cache (Z) from the forward propagation in the same layer.

    Returns:
    np.ndarray: The gradient of the cost with respect to Z.
    """
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA: np.ndarray, activation_cache: np.ndarray) -> np.ndarray:
    """
    Compute the backward propagation for a single sigmoid unit.

    Args:
    dA (np.ndarray): The post-activation gradient.
    activation_cache (np.ndarray): The activation cache (Z) from the forward propagation in the same layer.

    Returns:
    np.ndarray: The gradient of the cost with respect to Z.
    """
    Z = activation_cache
    S = 1 / (1 + np.exp(-Z))
    dZ = dA * S * (1 - S)
    return dZ