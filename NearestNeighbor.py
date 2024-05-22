import numpy as np

class NearestNeighbor:
    def __init__(self):
        self.training_data = None
        self.training_labels = None

    def train(self, data, labels):
        self.training_data = data
        self.training_labels = labels

    def test(self, instance):
        distances = np.linalg.norm(self.training_data - instance, axis=1)
        nearest_index = np.argmin(distances)
        return self.training_labels[nearest_index]