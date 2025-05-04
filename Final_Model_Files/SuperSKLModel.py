from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

from Plot import plotCM, plotP






class SuperSklearn:
    '''
    SuperClass for creating/loading and using supervised machine learning models from Scikit-Learn for making predictions and evaluating performance



    Attributes:
        model            - The loaded Scikit-Learn model                                              - sklearn model (created in subclasses)
        threshold        - The decision/cutoff boundary (likelihood of fraud that is flagged as such) - float (created in subclasses)
        scalar           - The scalar used to scale any data                                          - preprocessing object
        y_pred           - List to store predicted labels                                             - list
        y_true           - List to store true labels for evaluation                                   - list
        testing          - Flag to track if model is currently being tested on                        - bool
        features         - Features for input data                                                    - ndarray
        target           - Target labels for input data                                               - ndarray
        targetless       - Flag indicating if the input data does not have targets                    - bool
        X_file           - Features from the dataset file                                             - ndarray
        y_file           - Target labels from the dataset file                                        - ndarray
        cm               - A confusion matrix containing test results                                 - ndarray
        paddedcm         - Placeholder for cm to ensure there is no missing data                      - ndarray



    Methods:
        __init__ - Initializes the UseModelSklearn class by loading the model from a file
        
        resetStored - Resets the stored prediction and true labels lists
        
        loadTargetless - Loads input data for targetless predictions
        
            Parameters:
                data_points - Collection of entries relating to the dataset the model is trained on without targets - list of lists
        
        loadTargeted - Loads labeled input data for prediction
            
            Parameters:
                data_points - Collection of entries relating to the dataset the model is trained on with targets - list of lists
                existScalar - Indicates whether or not a scalar already exists                                   - bool
        
        predict - Makes predictions on the provided input data (either targetless or labeled)
        
        test - Tests the model on the loaded input data (if labeled), calculating accuracy, f1 score (weighted and unweighted), and confusion matrix
    '''
    
    def __init__(self):
        self.resetStored()
        self.testing = False
        
        self.features = None 
        self.target = None
        
        self.targetless = True
            
    def resetStored(self):
        self.y_pred = []
        self.y_true = []
        
    def loadTargetless(self, data_points):  # Ensure that input is a 2D array (each row is a separate data point)
        self.features = np.array(data_points) 
        self.features = self.scalar.transform(self.features) 
        self.targetless = False
        
    def loadTargeted(self, data_points, existScalar = True):  # Ensure that input is a 2D array (each row is a separate data point)
        self.data_points = np.array(data_points)
        self.features = self.data_points[:, :-1]
        if existScalar:
            self.features = self.scalar.transform(self.features)
        else:
            self.features = self.scalar.fit_transform(self.features)
        self.target = self.data_points[:, -1]
        self.targetless = False
        
    def predict(self):
        if self.features is None:
            pass 
        else:
            if not self.testing:
                self.resetStored()
                
            y_prob = self.model.predict_proba(self.features)[:, 1]  
            self.y_pred.extend((y_prob >= self.threshold).astype(int))
            
          
            if not self.testing:
                self.cm = confusion_matrix([0]*len(self.y_pred), self.y_pred)
                if self.cm.shape[1] <2:
                    self.padded_cm = np.zeros((1, 2), dtype=int)  
                    self.padded_cm[0, int(self.y_pred[0])] = self.cm[0, 0]     
                    self.cm = self.padded_cm
                plotP(self.cm)
                self.resetStored()
    
    def test(self):
        if self.targetless:
            pass
        else:
            self.testing = True
            
            self.predict()
            self.y_true.extend(self.target)
            self.cm = confusion_matrix(self.y_true, self.y_pred)
            if self.cm.shape[0] < 2 or self.cm.shape[1] < 2:
                self.padded_cm = np.zeros((2, 2), dtype=int)
            
                if self.cm.shape == (1, 1):
                    self.padded_cm[int(self.y_true[0]), int(self.y_pred[0])] = self.cm[0, 0]  
                elif self.cm.shape == (1, 2):  
                    self.padded_cm[int(self.y_true[0]), 0] = self.cm[0, 0]
                    self.padded_cm[int(self.y_true[0]), 1] = self.cm[0, 1]
                elif self.cm.shape == (2, 1): 
                    self.padded_cm[0, int(self.y_pred[0])] = self.cm[0, 0]
                    self.padded_cm[1, int(self.y_pred[0])] = self.cm[1, 0]
                
                self.cm = self.padded_cm
            self.testing = False
            
            print(f"Accuracy: {((self.cm[0, 0] + self.cm[1, 1]) / self.cm.sum())* 100:.2f}%")
            print(f"F1 Score: {(f1_score(self.y_true, self.y_pred)):.4f}")
            print(f"weighted F1 score: {(f1_score(self.y_true, self.y_pred, average='weighted')):.4f}")
            
            plotCM([self.cm], ["Model Test Results"])
    
    