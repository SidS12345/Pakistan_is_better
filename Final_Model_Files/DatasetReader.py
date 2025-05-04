import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import numpy as np






class DatasetReader(Dataset):
    '''
    Class for reading and processing a dataset - Inherits from the Dataset class



    Attributes:
        data         - The raw data loaded from the CSV file                                                  - DataFrame
        transform    - A transformation to apply to the features (e.g., scaling, normalization)               - callable (or None)
        features     - The features extracted from the dataset (all columns except the last one)              - ndarray
        target       - The target labels extracted from the dataset (last column)                             - ndarray
        undersampler - An instance of RandomUnderSampler used for balancing the dataset (if undersample=True) - RandomUnderSampler (or None)



    Methods:
        __init__ - Initializes the DatasetReader with the provided dataset file and optional transformation settings
        
            Parameters:
                csv_file    - The path to the CSV file containing the dataset                                - str
                undersample - A flag to indicate whether undersampling should be applied for class balancing - bool (optional)
                transform   - A transformation to apply to the features before returning them                - callable (optional)

        __len__ - Returns the size of the dataset (number of samples)

        size - Returns the number of features (columns) in the dataset

        __getitem__ - Retrieves a specific data sample at the provided index and returns both the features and target
        
            Parameters:
                idx - The index of the sample to retrieve - int
        '''
        
    def __init__(self, csv_file, undersample=False, scalar=None): 
        self.data = pd.read_csv(csv_file)
        
        self.features = self.data.iloc[:, :-1].values  # All columns except for the last one
        self.target = self.data.iloc[:, -1].values  # Just the last column
        
        self.scalar = scalar
        
        if undersample:
            self.undersampler = RandomUnderSampler(sampling_strategy='auto')
            self.features, self.target = self.undersampler.fit_resample(self.features, self.target)
            
        if self.scalar == None:
            self.scalar = StandardScaler()
            self.features = self.scalar.fit_transform(self.features) 
        else:
            self.features = self.scalar.transform(self.features)
            
    def __len__(self):
        return len(self.features)
    
    def size(self): 
        return self.features.shape[1]

    def __getitem__(self, idx): 
        sample_features = torch.tensor(self.features[idx], dtype=torch.float32)
        sample_target = torch.tensor(self.target[idx], dtype=torch.float32)

        return sample_features, sample_target
    
    def getScalar(self):
        return self.scalar
    

        
class TargetlessDatasetReader(DatasetReader):
    '''
    Class for accessing a database, standardizing the data, and preparing data for model input - Inherits from DatasetReader



    Attributes:
        data      - The raw data loaded from the CSV file                                    - DataFrame
        transform - A transformation to apply to the features (e.g., scaling, normalization) - callable (or None)
        features  - The features extracted from the dataset                                  - ndarray



    Methods:
        __init__ - Initializes the TargetlessDatasetReader with the provided dataset file and optional transformation settings
        
            Parameters:
                csv_file - The path to the CSV file containing the dataset                  - str
                transform - A transformation to apply to the features before returning them - callable (optional)

        __getitem__ - Retrieves a specific data sample at the provided index and returns the features
        
            Parameters:
                idx - The index of the sample to retrieve - int
    '''
    
    def __init__(self, csv_file, transform=None, scalar=None):
        self.data = pd.read_csv(csv_file)
        self.scalar = scalar
        
        
        self.features = self.data.iloc[:, :].values  
        
        if self.scalar == None:
            self.scalar = StandardScaler()
            self.features = self.scalar.fit_transform(self.features) 
        else:
            self.features = self.scalar.transform(self.features)
    
    def __getitem__(self, idx):
        sample_features = torch.tensor(self.features[idx], dtype=torch.float32)

        return sample_features