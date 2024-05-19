class Node:
    def __init__(self, subset=None, parent=None, accuracy=0):
        self.subset = subset or []
        self.parent = parent
        self.accuracy = accuracy

    def __lt__(self, other):
        return self.accuracy > other.accuracy




# class Node:
#     def __init__(self, num=0, state=None, parent=None, subset=None, accuracy=0, f_list=[]):
#         self.num = num 
#         self.state = state
#         self.parent = parent
#         self.subset = subset or []
#         self.accuracy = accuracy
#         self.feature_list = f_list
#         if parent:
#             self.depth = parent.depth + 1
#         else:
#             self.depth = 0

#     def __repr__(self):
#         return f"Node(subset={self.subset}, accuracy={self.accuracy})"

#     def __lt__(self, other):
#         return self.accuracy > other.accuracy



# # Ensure Node class has the correct structure
# class Node:
#     def __init__(self, state, parent, subset, accuracy):
#         self.state = state
#         self.parent = parent
#         self.subset = subset
#         self.accuracy = accuracy

#     def __lt__(self, other):
#         return self.accuracy < other.accuracy
    

