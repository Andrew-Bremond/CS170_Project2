import numpy as np
from ForwardSelection import ForwardSelection
from BackwardsSelection import BackwardSelection
from Bertie import Bertie

def load_data(file_path):
    data = np.loadtxt(file_path)
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    return features, labels

def beginningInput():
    print("Welcome to Bertie Wooster's Feature Selection Algorithm.")
    total_features = int(input("Please enter total number of features: "))
    print("Type the number of the algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    print("3. Bertie's Special Algorithm.")
    
    choice = input("Enter your choice of algorithm: 1, 2, or 3 \n")

    return total_features, choice 

def main():
    total_features, choice = beginningInput()

    # Load the data
    data_matrix, labels = load_data('large-test-dataset-1.txt')  # or 'large-test-dataset.txt'

    if choice == '1': 
        forward = ForwardSelection(total_features)
        forward.data_matrix = data_matrix
        forward.labels = labels
        forward.solve_forward_selection()
    elif choice == '2':
        backward = BackwardSelection(total_features)
        backward.data_matrix = data_matrix
        backward.labels = labels
        backward.solve_backward_selection()
    elif choice == '3':
        bertie = Bertie(total_features, data_matrix)
        bertie.labels = labels
        bertie.solve_bertie()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()