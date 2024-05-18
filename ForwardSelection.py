import random
from Node import Node
import heapq

class ForwardSelection:
    def __init__(self, num_features):
        self.num_features = num_features
        self.best_accuracy = 0
        self.best_features = set()

    def calculate_accuracy(self):
        return random.uniform(0, 1) * 100

    def solve_forward_selection(self):
        initial_node = Node(state=set(), parent=None, subset=[], accuracy=self.calculate_accuracy())
        print(f'Using no features and “random” evaluation, I get an accuracy of {initial_node.accuracy:.1f}%')
        print('Beginning search.')
        
        frontier = []
        heapq.heappush(frontier, (0, initial_node))
        best_node = initial_node

        while frontier:
            _, current = heapq.heappop(frontier)
            if len(current.subset) == self.num_features:
                if current.accuracy > self.best_accuracy:
                    self.best_accuracy = current.accuracy
                    self.best_features = set(current.subset)
                continue

            for feature in range(1, self.num_features + 1):
                if feature not in current.subset:
                    new_subset = current.subset + [feature]
                    accuracy = self.calculate_accuracy()
                    new_node = Node(state=set(new_subset), parent=current, subset=new_subset, accuracy=accuracy)
                    heapq.heappush(frontier, (-accuracy, new_node))
                    
                    print(f'Using feature(s) {set(new_subset)} accuracy is {accuracy:.1f}%')
                    if accuracy > self.best_accuracy:
                        self.best_accuracy = accuracy
                        self.best_features = set(new_subset)
                        print(f'Feature set {self.best_features} was best, accuracy is {self.best_accuracy:.1f}%')

        print('Finished search!!')
        print(f'The best feature subset is {self.best_features}, which has an accuracy of {self.best_accuracy:.1f}%')
        return self.sol_path(best_node)

    def sol_path(self, node):
        path = []
        accuracies = []
        current = node
        while current:
            path.append(current.subset)
            accuracies.append(current.accuracy)
            current = current.parent
        return path[::-1], accuracies[::-1]
