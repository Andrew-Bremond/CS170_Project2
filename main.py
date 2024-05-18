from ForwardSelection import ForwardSelection

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
        print("Backward Elimination is not yet implemented.")
    elif choice == '3':
        print("Bertie's Special Algorithm is not yet implemented.")
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
