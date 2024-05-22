import random
import heapq
import numpy as np
from Node import Node  # Import the Node class
from Validator import Validator

class Bertie:
    def __init__(self, num_features, data_matrix):
        self.num_features = num_features
        self.data_matrix = data_matrix
        self.best_accuracy = 0
        self.best_features = list(range(1, num_features + 1))
        self.previous_accuracy = 0  # Track the previous accuracy to check for warnings
        self.visited = set()  # Set to track visited nodes

    def calculate_accuracy(self):
        # Simulating accuracy calculation with a random value
        return random.uniform(0, 1) * 100

    def svd_feature_selection(self, features):
        # Perform Singular Value Decomposition (SVD) on the data matrix
        U, s, Vt = np.linalg.svd(self.data_matrix[:, features], full_matrices=False)
        # Select features based on the highest singular values
        selected_features = np.argsort(s)[::-1][:self.num_features]
        return [features[i] for i in selected_features]

    def solve_bertie(self):
        initial_node = Node(subset=self.best_features, accuracy=0)
        print('Using all features and “random” evaluation, I get an accuracy of 0%')
        print('Beginning search.')

        frontier = []
        heapq.heappush(frontier, (0, initial_node))
        self.visited.add(frozenset(initial_node.subset))
        
        while frontier:
            _, current = heapq.heappop(frontier)

            if len(current.subset) == 0:
                continue

            for feature in current.subset:
                new_subset = [f for f in current.subset if f != feature]
                new_subset_frozenset = frozenset(new_subset)

                if new_subset_frozenset in self.visited:
                    continue

                selected_features = self.svd_feature_selection(new_subset)
                validator = Validator(self.data_matrix, self.labels)
                accuracy = validator.leave_one_out_cross_validation(selected_features)
                new_node = Node(subset=selected_features, parent=current, accuracy=accuracy)
                heapq.heappush(frontier, (-accuracy, new_node))
                self.visited.add(new_subset_frozenset)

                feature_set = set(selected_features)
                print(f'Using feature(s) {feature_set} accuracy is {accuracy:.2f}%')
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_features = selected_features
                    print(f'Feature set {set(self.best_features)} was best, accuracy is {self.best_accuracy:.2f}%')

        print('Finished search!!')
        print(f'The best feature subset is {set(self.best_features)}, which has an accuracy of {self.best_accuracy:.2f}%')
        return self.get_solution_path()

    def get_solution_path(self):
        path, accuracies = [], []
        current = Node(subset=self.best_features, accuracy=self.best_accuracy)

        while current:
            path.append(current.subset)
            accuracies.append(current.accuracy)
            current = current.parent

        return path[::-1], accuracies[::-1]