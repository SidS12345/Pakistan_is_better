import joblib

from SuperSKLModel import SuperSklearn as SuperSKLM
from CSVToArray import toList






class UseModel(SuperSKLM):
    '''
    Class for loading and using supervised machine learning models from Scikit-Learn for making predictions and evaluating performance



    Attributes:
        model            - The loaded Scikit-Learn model                                              - sklearn model
        threshold        - The decision/cutoff boundary (likelihood of fraud that is flagged as such) - float
        scalar           - The scalar used to scale any data                                          - preprocessing object
        y_pred           - List to store predicted labels                                             - list
        y_true           - List to store true labels for evaluation                                   - list
        testing          - Flag to track if model is currently being tested on                        - bool
        input_features   - Features for input data                                                    - ndarray
        input_target     - Target labels for input data                                               - ndarray
        input_targetless - Flag indicating if the input data does not have targets                    - bool
        X_file           - Features from the dataset file                                             - ndarray
        y_file           - Target labels from the dataset file                                        - ndarray
        cm               - A confusion matrix containing test results                                 - ndarray
        paddedcm         - Placeholder for cm to ensure there is no missing data                      - ndarray
        accuracy         - Calculated accuracy of the model on the testing data                       - float
        model_data       - Loaded data from the joblib file                                           - dict
        
        
        
    Methods:
        __init__ - Initializes the UseModelSklearn class by loading the model from a file
        
            Parameters:
                model_file -  Path to the model file - str
        
        loadModel - Loads the trained model and relevant metadata (threshold, model type)
        
            Parameters:
                model_file -  Path to the model file (.pth for PyTorch) - str
        
        setThreshold - Allows setting a new threshold for classification
        
            Parameters:
                threshold - The decision/cutoff boundary (likelihood of fraud that is flagged as such) - float
        
        resetStored - Resets the stored prediction and true labels lists
        
        loadTargetless - Loads input data for targetless predictions
        
            Parameters:
                Data_points - Collection of entries relating to the dataset the model is trained on without targets - list of lists
        
        loadTargeted - Loads labeled input data for prediction
        
            Parameters:
                Data_points - Collection of entries relating to the dataset the model is trained on with targets - list of lists
        
        predict - Makes predictions on the provided input data (either targetless or labeled)
        
        test - Tests the model on the loaded input data (if labeled), calculating accuracy, f1 score (weighted and unweighted), and confusion matrix
    '''
    
    def __init__(self, model_file):
        super().__init__()
        self.loadModel(model_file)
        
    def loadModel(self, model_file):
        self.model_data = joblib.load(model_file)
        self.model = self.model_data["model"]
        

        self.threshold =  self.model_data["threshold"]
        self.scalar = self.model_data["scalar"]
        
    def setThreshold(self, threshold):
        self.threshold = threshold
            
    def loadTargeted(self, data_points):  # Ensure that input is a 2D array (each row is a separate data point)
        super().loadTargeted(data_points)





    
def main(): # Example code
    test_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/fraudTest.csv"
    data = toList(test_file) # toList only necessary if input is a csv file
    model = UseModel("RFModel_20250411_164240.joblib")
    model.loadTargeted(data)
    model.test()
    model.test()
    model.resetStored()
    model.setThreshold(0.5)
    model.test()
    




if __name__ == "__main__":
    main()