import random
import heapq
import numpy as np
from Node import Node

class Bertie:
    def __init__(self, data_matrix, labels, validator):
        self.data_matrix = data_matrix
        self.labels = labels
        self.validator = validator
        self.num_features = data_matrix.shape[1]
        self.best_accuracy = 0
        self.best_features = list(range(1, self.num_features + 1))
        self.previous_accuracy = 0  # Track the previous accuracy to check for warnings
        self.visited = set()  # Set to track visited nodes

    def calculate_accuracy(self, features):
        # Use validator to calculate accuracy
        return self.validator.leave_one_out_cross_validation(features) * 100

    def svd_feature_selection(self, features):
        zero_based_features = [f - 1 for f in features]
        U, s, Vt = np.linalg.svd(self.data_matrix[:, zero_based_features], full_matrices=False)
        selected_features = np.argsort(s)[::-1][:len(features)]
        return [features[i] for i in selected_features]

    def solve_bertie(self):
        initial_node = Node(subset=self.best_features, accuracy=self.calculate_accuracy(self.best_features))
        print(f'Using all features and "random" evaluation, I get an accuracy of {initial_node.accuracy:.1f}%')
        print('Beginning search.')
        frontier = []
        heapq.heappush(frontier, (-initial_node.accuracy, initial_node))
        self.visited.add(frozenset(initial_node.subset))
        self.best_accuracy = initial_node.accuracy

        while frontier:
            _, current = heapq.heappop(frontier)
            if len(current.subset) == 0:
                continue
            for feature in current.subset:
                new_subset = [f for f in current.subset if f != feature]
                new_subset_frozenset = frozenset(new_subset)
                if new_subset_frozenset in self.visited:
                    continue
                # Perform SVD-based feature selection
                selected_features = self.svd_feature_selection(new_subset)
                accuracy = self.calculate_accuracy(selected_features)
                new_node = Node(subset=selected_features, parent=current, accuracy=accuracy)
                heapq.heappush(frontier, (-accuracy, new_node))
                self.visited.add(new_subset_frozenset)
                feature_set = set(selected_features)
                print(f'Using feature(s) {feature_set} accuracy is {accuracy:.1f}%')
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_features = selected_features
                    print(f'Feature set {set(self.best_features)} was best, accuracy is {self.best_accuracy:.1f}%')
                # elif accuracy < self.previous_accuracy:
                #     print('(Warning, Accuracy has decreased!)')
                self.previous_accuracy = accuracy

        print('Finished search!!')
        print(f'The best feature subset is {set(self.best_features)}, which has an accuracy of {self.best_accuracy:.1f}%')
        return self.get_solution_path()

    def get_solution_path(self):
        path, accuracies = [], []
        current = Node(subset=self.best_features, accuracy=self.best_accuracy)
        while current:
            path.append(current.subset)
            accuracies.append(current.accuracy)
            current = current.parent
        return path[::-1], accuracies[::-1]
