import random
import heapq
import time
from Node import Node 
from Validator import Validator

class BackwardSelection:
    def __init__(self, data_matrix, labels):
        self.data_matrix = data_matrix
        self.labels = labels
        self.best_accuracy = 0
        self.best_features = []

    def solve_backward_selection(self, features):
        initial_node = Node(subset=features, accuracy=0)
        validator = Validator(self.data_matrix, self.labels)
        initial_node.accuracy = validator.leave_one_out_cross_validation(features)
        print(f'Initial accuracy with all features {features}: {initial_node.accuracy:.2f}')

        frontier = []
        heapq.heappush(frontier, (0, initial_node))
        visited = set()
        visited.add(frozenset(initial_node.subset))

        while frontier:
            _, current = heapq.heappop(frontier)

            if len(current.subset) == 1:
                continue

            for feature in current.subset:
                new_subset = current.subset.copy()
                new_subset.remove(feature)
                new_subset_frozenset = frozenset(new_subset)

                if new_subset_frozenset in visited:
                    continue

                accuracy = validator.leave_one_out_cross_validation(new_subset)
                new_node = Node(subset=new_subset, parent=current, accuracy=accuracy)
                heapq.heappush(frontier, (-accuracy, new_node))
                visited.add(new_subset_frozenset)

                feature_set = set(new_subset)
                print(f'Using feature(s) {feature_set} accuracy is {accuracy:.2f}')
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_features = new_subset
                    print(f'Feature set {set(self.best_features)} was best, accuracy is {self.best_accuracy:.2f}')

        end_time = time.time()
        print('Finished search!!')
        print(f'The best feature subset is {set(self.best_features)}, which has an accuracy of {self.best_accuracy:.2f}')
