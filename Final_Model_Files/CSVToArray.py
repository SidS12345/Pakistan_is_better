import csv
from imblearn.under_sampling import RandomUnderSampler
import numpy as np






def toList(csv_file):
    '''
    !!!Not necessary for final code, used only for running the code on csv files!!!
    
    Function to convert a CSV file into a 2D array of the correct format
    
    
    
    parameters:
    
        csv_file - Path to a csv file to be converted - str
    '''
    
    data = []
    with open(csv_file, newline='') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            float_row = [float(val) for val in row]
            data.append(float_row)
    return data
    
    
    
def undersample(data):
    '''
    Function to take a 2D Array of data and undersample it for training
    
    Parameters:

        data - collection of entries to be trained on - list of lists
    '''
    
    data = np.array(data)
    X = data[:, :-1]  
    y = data[:, -1]   

    X_res, y_res = RandomUnderSampler().fit_resample(X, y)
    
    return np.hstack((X_res, y_res[:, None])).tolist()