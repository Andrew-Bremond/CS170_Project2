class Node:
    def __init__(self, subset=None, parent=None, accuracy=0):
        self.subset = subset or []
        self.parent = parent
        self.accuracy = accuracy

    def __lt__(self, other):
        return self.accuracy > other.accuracy
