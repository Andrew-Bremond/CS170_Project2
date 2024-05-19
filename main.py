import numpy as np
from ForwardSelection import ForwardSelection
from BackwardsSelection import BackwardSelection
from Bertie import Bertie

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

    num_samples = 100  # Number of samples Replace once data is available
    num_features = 10  # Number of features Replace once data is available
    data_matrix = np.random.rand(num_samples, num_features) # Random data matrix Replace once data is available
    
    if choice == '1': 
        forward = ForwardSelection(total_features)
        forward.solve_forward_selection()
    elif choice == '2':
        backwards = BackwardSelection(total_features)
        backwards.solve_backward_selection()
    elif choice == '3':
        bertie = Bertie(total_features, data_matrix)
        bertie.solve_bertie()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
