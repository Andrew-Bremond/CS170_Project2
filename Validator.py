import numpy as np
from NearestNeighbor import NearestNeighbor

class Validator:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def leave_one_out_cross_validation(self, feature_subset):
        correct_predictions = 0
        num_instances = len(self.labels)

        for i in range(num_instances):
            train_data = np.delete(self.data, i, axis=0)[:, feature_subset]
            train_labels = np.delete(self.labels, i, axis=0)
            test_instance = self.data[i, feature_subset]

            nn = NearestNeighbor()
            nn.train(train_data, train_labels)
            prediction = nn.test(test_instance)

            if prediction == self.labels[i]:
                correct_predictions += 1

        accuracy = correct_predictions / num_instances
        return accuracy