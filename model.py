import numpy as np
import copy
from typing import List, Tuple, Dict
from layers import linear_activation_forward, linear_activation_backward

class Neural_Network():
    def __init__(self, layers_dims: List[int]):
        """
        Initialize the neural network with the given layer dimensions.
        
        Parameters:
            layers_dims (list): List containing the dimensions of each layer in the network.
        """
        self.layers_dims = layers_dims
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self) -> Dict[str, np.ndarray]:
        """
        Initialize the parameters of the network with small random values for weights and zeros for biases.
        
        Returns:
            dict: Dictionary containing the initialized weights and biases.
        """
        np.random.seed(3)
        parameters = {}
        for l in range(1, len(self.layers_dims)):
            parameters[f'W{l}'] = np.random.randn(self.layers_dims[l], self.layers_dims[l-1]) * 0.01
            parameters[f'b{l}'] = np.zeros((self.layers_dims[l], 1))
        return parameters

    def train(self, X: np.ndarray, Y: np.ndarray, learning_rate: float = 0.0075, num_iterations: int = 3000, print_cost: bool = False) -> List[float]:
        """
        Train the neural network using the provided training data.
        
        Parameters:
            X (np.array): Input features, shape (number of features, number of examples).
            Y (np.array): True labels, shape (1, number of examples).
            learning_rate (float): Learning rate for the gradient descent update rule.
            num_iterations (int): Number of iterations to run the gradient descent.
            print_cost (bool): If True, print the cost every 100 iterations.
        
        Returns:
            list: List of costs computed during the training, useful for plotting learning curves.
        """
        np.random.seed(1)
        costs = []
        for i in range(num_iterations):
            AL, caches = self.L_model_forward(X)
            cost = self.compute_cost(AL, Y)
            grads = self.L_model_backward(AL, Y, caches)
            self.update_parameters(grads, learning_rate)
            if print_cost and (i % 100 == 0 or i == num_iterations - 1):
                print(f"Cost after iteration {i}: {np.squeeze(cost)}")
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)
        return costs

    def compute_cost(self, AL: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the cross-entropy cost function.
        
        Parameters:
            AL (np.array): Probability vector corresponding to label predictions, shape (1, number of examples).
            Y (np.array): True label vector, shape (1, number of examples).
        
        Returns:
            float: Cross-entropy cost.
        """
        m = Y.shape[1]
        cost = -1/m * np.sum((Y * np.log(AL)) + (1-Y) * np.log(1-AL))
        cost = np.squeeze(cost) 
        return cost

    def L_model_forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]]:
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation.
        
        Parameters:
            X (np.array): Data, numpy array of shape (input size, number of examples).
        
        Returns:
            tuple: The output of the last layer and the cache list containing all the caches.
        """
        caches = []
        A = X
        L = len(self.parameters) // 2
        for l in range(1, L):
            A_prev = A 
            A, cache = linear_activation_forward(A_prev, 
                                                self.parameters[f"W{l}"], 
                                                self.parameters[f"b{l}"], 
                                                activation='relu')
            caches.append(cache)
                    
        AL, cache = linear_activation_forward(A, 
                                              self.parameters[f"W{L}"], 
                                              self.parameters[f"b{L}"], 
                                              activation='sigmoid')
        caches.append(cache)
                    
        return AL, caches

    def L_model_backward(self, AL: np.ndarray, Y: np.ndarray, caches: List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]) -> Dict[str, np.ndarray]:
        """
        Implement the backward propagation for the entire network.
        
        Parameters:
            AL (np.array): Probability vector, output of the forward propagation (L_model_forward()).
            Y (np.array): True label vector.
            caches (list): List of caches containing every cache of linear_activation_forward().
        
        Returns:
            dict: A dictionary with the gradients.
        """
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[-1]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation="sigmoid")
        grads[f"dA{L-1}"] = dA_prev_temp
        grads[f"dW{L}"] = dW_temp
        grads[f"db{L}"] = db_temp
        
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, activation="relu")
            grads[f"dA{l}"] = dA_prev_temp
            grads[f"dW{l+1}"] = dW_temp
            grads[f"db{l+1}"] = db_temp
            
        return grads

    def update_parameters(self, grads: Dict[str, np.ndarray], learning_rate: float):
        """
        Update parameters using gradient descent.
        
        Parameters:
            grads (dict): Dictionary containing the gradients.
            learning_rate (float): Learning rate.
        """
        L = len(self.parameters) // 2
        parameters_copy = copy.deepcopy(self.parameters)  # Create a deep copy of parameters
        for l in range(L):
            parameters_copy[f'W{l+1}'] -= learning_rate * grads[f'dW{l+1}']
            parameters_copy[f'b{l+1}'] -= learning_rate * grads[f'db{l+1}']
        self.parameters = parameters_copy

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the results using a trained neural network.
        
        Parameters:
            X (np.array): Input data, shape (number of features, number of examples).
        
        Returns:
            np.array: Predictions (0/1) for the input dataset.
        """
        m = X.shape[1]
        p = np.zeros((1, m), dtype=int)
        probas, caches = self.L_model_forward(X)
        for i in range(probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        return p

    def accuracy(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        """
        Calculate the accuracy of predictions.
        
        Parameters:
            Y_pred (np.array): Predicted labels.
            Y_true (np.array): True labels.
        
        Returns:
            float: Accuracy percentage.
        """
        return np.mean(Y_pred == Y_true) * 100