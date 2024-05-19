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
    initial_state, choice = beginningInput()
    if choice == '1': 
        forward = ForwardSelection(initial_state)
        forward.solve_forward_selection()
    elif choice == '2':
        backwards = BackwardSelection(initial_state)
        backwards.solve_backward_selection()
    elif choice == '3':
        bertie = Bertie(initial_state)
        bertie.solve_bertie()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
