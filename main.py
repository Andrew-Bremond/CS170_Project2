import numpy as np
import time
from ForwardSelection import ForwardSelection
from BackwardsSelection import BackwardSelection
from Bertie import Bertie
from Classifier import Classifier
from Validator import Validator

def load_data(file_path="small-test-dataset-1.txt"):
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

    # normalize
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    if np.any(std == 0):
        print("Warning: Some features have zero standard deviation.")
    std[std == 0] = 1 
    data = (data - mean) / std

    return data, labels

def beginning_input():
    print("Welcome to Bertie Wooster's Feature Selection Algorithm! by jgonz671, abrem005, and dcoel003.")
    print("Type the number of the algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    print("3. Bertie's Special Algorithm.")
    
    choice = input("Enter your choice of algorithm: 1, 2, or 3 \n")
    features = list(map(int, input("Enter the list of features separated by spaces: ").split()))

    return choice, features

def main():
    start_time = time.time()
    choice, features = beginning_input()

    features = [f-1 for f in features]  # index data from 0 

    data_matrix, labels = load_data('large-test-dataset-1.txt')  # use large or small dataset
    classifier = Classifier()
    validator = Validator(classifier, data_matrix, labels)
    accuracy = validator.leave_one_out_cross_validation(features)
    features = [f+1 for f in features]
    print(f'Accuracy using features {features}: {accuracy:.2f}')

    end_time = time.time()  
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()
