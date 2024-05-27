import heapq
import time
from Node import Node
from Validator import Validator

class ForwardSelection:
    def __init__(self, data_matrix, labels):
        self.data_matrix = data_matrix
        self.labels = labels
        self.best_accuracy = 0
        self.best_features = []
        self.visited = set()

    def solve_forward_selection(self, features):
        initial_node = Node(subset=[], accuracy=0)
        print('Using no features and “random” evaluation, I get an accuracy of 0')
        print('Beginning search.')

        frontier = []
        heapq.heappush(frontier, (0, initial_node))
        self.visited.add(frozenset(initial_node.subset))
        
        while frontier:
            _, current = heapq.heappop(frontier)

            if len(current.subset) == len(features):
                continue

            for feature in features:
                if feature not in current.subset:
                    new_subset = current.subset + [feature]
                    new_subset_frozenset = frozenset(new_subset)

                    if new_subset_frozenset in self.visited:
                        continue

                    validator = Validator(self.data_matrix, self.labels)
                    accuracy = validator.leave_one_out_cross_validation(new_subset)
                    new_node = Node(subset=new_subset, parent=current, accuracy=accuracy)
                    heapq.heappush(frontier, (-accuracy, new_node))
                    self.visited.add(new_subset_frozenset)

                    feature_set = set(new_subset)
                    print(f'Using feature(s) {feature_set} accuracy is {accuracy:.2f}')
                    if (accuracy > self.best_accuracy):
                        self.best_accuracy = accuracy
                        self.best_features = new_subset
                        print(f'Feature set {set(self.best_features)} was best, accuracy is {self.best_accuracy:.2f}')

        print('Finished search!!')
        print(f'The best feature subset is {set(self.best_features)}, which has an accuracy of {self.best_accuracy:.2f}')
        return self.get_solution_path()

    def get_solution_path(self):
        path, accuracies = [], []
        current = Node(subset=self.best_features, accuracy=self.best_accuracy)

        while current:
            path.append(current.subset)
            accuracies.append(current.accuracy)
            current = current.parent

        return path[::-1], accuracies[::-1]
