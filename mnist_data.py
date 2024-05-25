from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List

class MNIST:
    def __init__(self):
        """Initialize the MNIST data class, preparing and normalizing data."""
        self.x_train, self.y_train, self.x_test, self.y_test = self.prepare_data()
        self.normalize_images()

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare the data by loading, filtering, and reshaping.

        Returns:
            A tuple containing training images, training labels, testing images, and testing labels.
        """
        x_train, y_train, x_test, y_test = self.load_data()
        x_train, y_train = self.filter_digits(x_train, y_train)
        x_test, y_test = self.filter_digits(x_test, y_test)
        x_train, y_train = self.reshape_data(x_train, y_train)
        x_test, y_test = self.reshape_data(x_test, y_test)
        return x_train, y_train, x_test, y_test

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the full MNIST dataset from keras.datasets.

        Returns:
            Tuple containing training images, training labels, testing images, and testing labels.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return x_train, y_train, x_test, y_test

    def filter_digits(self, images: np.ndarray, labels: np.ndarray, digits: Tuple[int, ...] = (0, 1)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter images and labels to only keep the specified digits.

        Args:
            images: The array of images.
            labels: The array of labels corresponding to the images.
            digits: A tuple of digit classes to retain.

        Returns:
            A tuple of filtered images and labels.
        """
        filter_mask = np.isin(labels, digits)
        filtered_images = images[filter_mask]
        filtered_labels = labels[filter_mask]
        return filtered_images, filtered_labels

    def reshape_data(self, images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape images to 2D and labels to 1D.

        Args:
            images: The array of images.
            labels: The array of labels.

        Returns:
            A tuple of reshaped images and labels.
        """
        images = images.reshape(images.shape[0], -1).T
        labels = labels.reshape(-1, 1).T
        return images, labels

    def normalize_images(self):
        """Normalize image data to 0-1 range."""
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

    def show_img(self, image_index: int):
        """
        Display an image by index from the test set.

        Args:
            image_index: The index of the image in the test set to display.

        Raises:
            ValueError: If the image index is out of range.
        """
        if image_index < 0 or image_index >= self.x_test.shape[1]:
            raise ValueError("Image index out of range")
        selected_image = self.x_test[:, image_index].reshape(28, 28)
        plt.imshow(selected_image, cmap="gray")
        plt.title(f"Label: {self.y_test[0, image_index]}")
        plt.show()