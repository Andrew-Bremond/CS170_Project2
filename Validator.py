import numpy as np

class Validator:
    def __init__(self, classifier, data_matrix, labels):
        self.classifier = classifier
        self.data_matrix = data_matrix
        self.labels = labels

    def leave_one_out_cross_validation(self, features):
        correct_predictions = 0
        num_instances = len(self.data_matrix)

        for i in range(num_instances):
            training_data_subset = np.delete(self.data_matrix, i, axis=0)
            training_labels_subset = np.delete(self.labels, i)
            test_instance = self.data_matrix[i, features]
            self.classifier.train(training_data_subset[:, features], training_labels_subset)
            predicted_label = self.classifier.test(test_instance)
            if predicted_label == self.labels[i]:
                correct_predictions += 1

        accuracy = correct_predictions / num_instances
        return accuracy
