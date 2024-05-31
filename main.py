import numpy as np
import heapq
import time
from ForwardSelection import ForwardSelection
from BackwardsSelection import BackwardSelection
from Bertie import Bertie
from Classifier import Classifier
from Validator import Validator
from Node import Node  # Ensure Node.py is also available

def load_data(file_path):
    data = []
    labels = []

    with open(file_path, 'r') as file:
        for row in file:
            row = row.strip().split()
            if row:
                try:
                    class_val = int(float(row[0]))
                    feature_vals = [float(val) for val in row[1:]]
                    labels.append(class_val)
                    data.append(feature_vals)
                except ValueError as e:
                    print(f"Error parsing row {row}: {e}")
                    continue

    data = np.array(data)
    labels = np.array(labels)
    if data.size == 0:
        raise ValueError("No data loaded. Please check the input file.")

    # Normalize the data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    if np.any(std == 0):
        print("Warning: Some features have zero standard deviation.")
    std[std == 0] = 1
    data = (data - mean) / std

    return data, labels

def select_file():
    print("Select the dataset file to use:")
    print("1. small-test-dataset-1.txt")
    print("2. large-test-dataset-1.txt")
    print("3. CS170_Spring_2024_Small_data__39.txt")
    print("4. CS170_Spring_2024_Large_data__39.txt")
    
    choice = input("Enter your choice (1, 2, 3, or 4): ")
    file_mapping = {
        "1": "small-test-dataset-1.txt",
        "2": "large-test-dataset-1.txt",
        "3": "CS170_Spring_2024_Small_data__39.txt",
        "4": "CS170_Spring_2024_Large_data__39.txt"
    }

    return file_mapping.get(choice, "small-test-dataset-1.txt")

def beginning_input():
    print("Welcome to Bertie Wooster's Feature Selection Algorithm! by jgonz671, abrem005, and dcoel003.")
    print("Type the number of the algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    print("3. Bertie's Special Algorithm.")
    
    choice = input("Enter your choice of algorithm: 1, 2, or 3 \n")
    return choice

def main():
    file_path = select_file()
    choice = beginning_input()

    data_matrix, labels = load_data(file_path)
    classifier = Classifier()
    validator = Validator(classifier, data_matrix, labels)
    features = list(range(data_matrix.shape[1]))  # Features indexed from 0 to n-1

    if choice == '1':
        forward = ForwardSelection(data_matrix, labels, validator)
        forward.solve_forward_selection(features)
    elif choice == '2':
        backward = BackwardSelection(data_matrix, labels, validator)
        backward.solve_backward_selection(features)
    elif choice == '3':
        bertie = Bertie(data_matrix, labels, validator)
        bertie.solve_bertie()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()