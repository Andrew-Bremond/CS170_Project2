import numpy as np
from sklearn.model_selection import StratifiedKFold

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
            # Convert 1-based feature indices to 0-based
            zero_based_features = [f - 1 for f in features]
            test_instance = self.data_matrix[i, zero_based_features]
            self.classifier.train(training_data_subset[:, zero_based_features], training_labels_subset)
            predicted_label = self.classifier.test(test_instance)
            if predicted_label == self.labels[i]:
                correct_predictions += 1

        accuracy = correct_predictions / num_instances
        return accuracy

    # def stratified_cross_validation(self, features, num_folds=5):
    #     skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    #     accuracies = []

    #     for train_index, test_index in skf.split(self.data_matrix, self.labels):
    #         training_data, test_data = self.data_matrix[train_index], self.data_matrix[test_index]
    #         training_labels, test_labels = self.labels[train_index], self.labels[test_index]

    #         # Convert 1-based feature indices to 0-based
    #         zero_based_features = [f - 1 for f in features]

    #         self.classifier.train(training_data[:, zero_based_features], training_labels)
    #         fold_accuracy = self._evaluate_accuracy(test_data, test_labels, zero_based_features)
    #         accuracies.append(fold_accuracy)

    #     mean_accuracy = np.mean(accuracies)
    #     return mean_accuracy

    # def _evaluate_accuracy(self, test_data, test_labels, features):
    #     correct_predictions = 0
    #     num_instances = len(test_data)

    #     for i in range(num_instances):
    #         test_instance = test_data[i, features]
    #         predicted_label = self.classifier.test(test_instance)
    #         if predicted_label == test_labels[i]:
    #             correct_predictions += 1

    #     accuracy = correct_predictions / num_instances
    #     return accuracy
