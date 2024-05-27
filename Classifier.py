import numpy as np

class Classifier:
    def __init__(self):
        self.training_data = None
        self.training_labels = None

    def train(self, data_matrix, labels):
        self.training_data = data_matrix
        self.training_labels = labels

    def test(self, test_instance):
        if self.training_data is None or self.training_labels is None:
            raise ValueError("Classifier not trained yet. Call train() first.")
        distances = np.linalg.norm(self.training_data - test_instance, axis=1)
        nearest_neighbor_index = np.argmin(distances)
        return self.training_labels[nearest_neighbor_index]
