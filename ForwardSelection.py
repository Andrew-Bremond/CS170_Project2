import random
import heapq
from Node import Node  # Import the Node class

class ForwardSelection:
    def __init__(self, num_features):
        self.num_features = num_features
        self.best_accuracy = 0
        self.best_features = []
        self.previous_accuracy = 0  # Track the previous accuracy to check for warnings
        self.visited = set()  # Set to track visited nodes

    def calculate_accuracy(self):
        # Simulating accuracy calculation with a random value
        return random.uniform(0, 1) * 100

    def solve_forward_selection(self):
        initial_node = Node(subset=[], accuracy=self.calculate_accuracy())
        print(f'Using no features and “random” evaluation, I get an accuracy of {initial_node.accuracy:.1f}%')
        print('Beginning search.')

        frontier = []
        heapq.heappush(frontier, (0, initial_node))
        self.visited.add(frozenset(initial_node.subset))
        
        while frontier:
            _, current = heapq.heappop(frontier)

            if len(current.subset) == self.num_features:
                if current.accuracy > self.best_accuracy:
                    self.best_accuracy = current.accuracy
                    self.best_features = current.subset
                continue

            for feature in range(1, self.num_features + 1):
                if feature not in current.subset:
                    new_subset = current.subset + [feature]
                    new_subset_frozenset = frozenset(new_subset)

                    if new_subset_frozenset in self.visited:
                        continue
                    
                    accuracy = self.calculate_accuracy()
                    new_node = Node(subset=new_subset, parent=current, accuracy=accuracy)
                    heapq.heappush(frontier, (-accuracy, new_node))
                    self.visited.add(new_subset_frozenset)

                    feature_set = set(new_subset)
                    print(f'Using feature(s) {feature_set} accuracy is {accuracy:.1f}%')
                    if accuracy > self.best_accuracy:
                        self.best_accuracy = accuracy
                        self.best_features = new_subset
                        print(f'Feature set {set(self.best_features)} was best, accuracy is {self.best_accuracy:.1f}%')
                    elif accuracy < self.previous_accuracy:
                        print('(Warning, Accuracy has decreased!)')

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


