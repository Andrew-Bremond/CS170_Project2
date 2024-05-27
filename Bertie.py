import random
import heapq
import numpy as np
import time
from Node import Node
from Validator import Validator

class Bertie:
    def __init__(self, data_matrix, labels):
        self.data_matrix = data_matrix
        self.labels = labels
        self.num_features = data_matrix.shape[1]
        self.best_accuracy = 0
        self.best_features = list(range(self.num_features))
        self.visited = set()

    def solve_bertie(self):
        start_time = time.time()
        
        initial_node = Node(subset=self.best_features, accuracy=0)
        validator = Validator(self.data_matrix, self.labels)
        initial_node.accuracy = validator.leave_one_out_cross_validation(self.best_features)
        print(f'Using all features and “random” evaluation, I get an accuracy of {initial_node.accuracy:.2f}%')
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

                accuracy = validator.leave_one_out_cross_validation(new_subset)
                new_node = Node(subset=new_subset, parent=current, accuracy=accuracy)
                heapq.heappush(frontier, (-accuracy, new_node))
                self.visited.add(new_subset_frozenset)

                feature_set = set(new_subset)
                print(f'Using feature(s) {feature_set} accuracy is {accuracy:.2f}%')
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_features = new_subset
                    print(f'Feature set {set(self.best_features)} was best, accuracy is {self.best_accuracy:.2f}%')

        end_time = time.time()
        print('Finished search!!')
        print(f'The best feature subset is {set(self.best_features)}, which has an accuracy of {self.best_accuracy:.2f}%')
        print(f'Execution time: {end_time - start_time:.2f} seconds')
        return self.get_solution_path()

    def get_solution_path(self):
        path, accuracies = [], []
        current = Node(subset=self.best_features, accuracy=self.best_accuracy)

        while current:
            path.append(current.subset)
            accuracies.append(current.accuracy)
            current = current.parent

        return path[::-1], accuracies[::-1]
