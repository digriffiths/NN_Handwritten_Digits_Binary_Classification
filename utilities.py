import numpy as np
import matplotlib.pyplot as plt

def plot_costs(costs: np.ndarray, learning_rate: float = 0.0075) -> None:
    """
    Plot the cost function values over iterations.

    Args:
    costs (np.ndarray): Array of cost values.
    learning_rate (float): The learning rate used in the training model.

    Returns:
    None
    """
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title(f"Learning rate = {learning_rate}")
    plt.show()

def find_misclassified(X: np.ndarray, Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """
    Identify the indices of misclassified samples.

    Args:
    X (np.ndarray): Input features, not used in this function but typically part of the signature for such functions.
    Y_true (np.ndarray): True labels.
    Y_pred (np.ndarray): Predicted labels.

    Returns:
    np.ndarray: Indices of misclassified samples.
    """
    misclassified_indices = np.where(Y_true != Y_pred)[1]
    return misclassified_indices