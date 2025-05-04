import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

from SuperCreateModel import SklearnModel as SKLM
from CSVToArray import toList, undersample





        
class NeuralNetwork(SKLM):
    '''
    Class for creating a Neural Network (Multilayer Perceptron) ML algorithm - Inherits from SklearnModel in the SuperModels file



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
        train_input      - The training data                                                          - list of lists
        test_input       - The testing data                                                           - list of lists
        model_type       - String for filename                                                        - str
        timestamp        - String for uniqueness of filename                                          - str
        filename         - String that the saved file will be named                                   - str
        to_save          - Data to be contained in saved file                                         - dict



    Methods:
        __init__ - Initializes the Neural Network model with given parameters and prepares the model for training
        
            Parameters:
                train_file         - location of dataset file to be used for training                                               - str
                test_file          - location of dataset file to be used for testing                                                - str
                threshold          - the decision/cutoff boundary (likelihood of fraud that is flagged as such)                     - float (optional)
                hidden_layer_sizes - the number of neurons in each hidden layer (tuple of integers)                                 - tuple (optional)
                activation_fn      - mathematical operation applied to output of a neuron - introduces non-linearity to the network - str (optional)
                solver             - the optimizer algorithm ('adam', 'sgd', etc.)                                                  - str (optional)
                alpha              - regularization term to prevent overfitting (L2 penalty)                                        - float (optional)
                batch_size         - number of samples per iteration before updating the models weights                             - str or int (optional)
                learning_rate      - how much model weights are adjusted wrt loss gradient - determines step size at each iteration - str (optional)
                learning_rate_init - initial learning rate for the solver                                                           - float (optional)
                max_iter           - maximum number of iterations for the solver to converge                                        - int (optional)
                early_stopping     - whether to stop training when validation score is not improving                                - bool (optional)
                momentum           - allows for faster convergence, but may overshoot                                               - float (optional)
                n_iter_no_change   - number of iterations with no improvement before stopping                                       - int (optional)
        
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
        
        train - Trains the model using the training dataset
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
        '''
        
    def __init__(self, train_file, test_file, threshold=0.5, 
                 hidden_layer_sizes=(4,5,2), activation_fn='relu', solver='adam',
                 alpha=0.0001, batch_size='auto', learning_rate='adaptive', 
                 learning_rate_init=0.003, max_iter=200, early_stopping=False,
                 momentum=0.9, n_iter_no_change=100):
        
        super().__init__(train_file, test_file, threshold)
        
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation_fn,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            momentum=momentum,
            n_iter_no_change=n_iter_no_change
        )
        
        self.model_type = "NN"

 

class LogisticRegressionModel(SKLM):
    '''
    Class for creating a Logistic Regression ML algorithm - Inherits from Sklearn in the SuperModels file
    
    
    
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
        train_input      - The training data                                                          - list of lists
        test_input       - The testing data                                                           - list of lists
        model_type       - String for filename                                                        - str
        timestamp        - String for uniqueness of filename                                          - str
        filename         - String that the saved file will be named                                   - str
        to_save          - Data to be contained in saved file                                         - dict



    Methods:
        __init__ - Initializes the Logistic Regression model with given parameters and prepares the model for training
                
            Parameters:
                    train_file    - Location of dataset file to be used for training                           - str
                    test_file     - Location of dataset file to be used for testing                            - str
                    threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such) - float (optional)
                    C             - Regularization strength; smaller values specify stronger regularization    - float (optional)
                    max_iter      - Maximum number of iterations for the solver to converge                    - int (optional)
                    solver        - Optimization algorithm                                                     - str (optional)
                    penalty       - Regularization type                                                        - str (optional)
                    tol           - Tolerance for stopping criteria (smaller values = stricter stopping)       - float (optional)
                    class_weight  - Weights associated with classes to handle imbalanced classes               - str or dict (optional)
                    fit_intercept - Whether to include an intercept in the model                               - bool (optional)
                
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
        
        train - Trains the model using the training dataset
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, threshold = 0.35, C=5e-3, max_iter=200, solver='saga', penalty='l1', tol=1e-8, class_weight='balanced', fit_intercept=False):
        super().__init__(train_file, test_file, threshold)
    
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter, 
            solver=solver,
            penalty=penalty, 
            tol=tol,
            class_weight=class_weight, 
            fit_intercept=fit_intercept 
        )
        
        self.model_type = "LR"
        


class NaiveBayes(SKLM):
    '''
    Class for creating a Naive Bayes ML algorithm - Inherits from Sklearn in the SuperModels file



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
        train_input      - The training data                                                          - list of lists
        test_input       - The testing data                                                           - list of lists
        model_type       - String for filename                                                        - str
        timestamp        - String for uniqueness of filename                                          - str
        filename         - String that the saved file will be named                                   - str
        to_save          - Data to be contained in saved file                                         - dict



    Methods:
        __init__ - Initializes the Naive Bayes model with given parameters and prepares the model for training
        
            Parameters:
                train_file    - Location of dataset file to be used for training                                  - str
                test_file     - Location of dataset file to be used for testing                                   - str
                threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)        - float (optional)
                var_smoothing - Variance smoothing parameter to prevent division by zero in Gaussian distribution - float (optional)
                priors        - The class prior probabilities (default assumes imbalanced classes)                - list of floats (optional)
        
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
        
        train - Trains the model using the training dataset
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, threshold=0.5, var_smoothing=1e-9, priors=[0.3,0.7]):
        super().__init__(train_file, test_file, threshold)
        
        self.model = GaussianNB(var_smoothing=var_smoothing, priors=priors)
        
        self.model_type = "NaiveBayes"
        


class GradientBoostingMachineModel(SKLM):
    '''
    Class for creating a Gradient Boosting ML algorithm - Inherits from Sklearn in the SuperModels file



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
        train_input      - The training data                                                          - list of lists
        test_input       - The testing data                                                           - list of lists
        model_type       - String for filename                                                        - str
        timestamp        - String for uniqueness of filename                                          - str
        filename         - String that the saved file will be named                                   - str
        to_save          - Data to be contained in saved file                                         - dict



    Methods:
        __init__ - Initializes the Gradient Boosting model with given parameters and prepares the model for training
        
            Parameters:
                train_file    - Location of dataset file to be used for training                                                                   - str
                test_file     - Location of dataset file to be used for testing                                                                    - str
                threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)                                         - float (optional)
                n_estimators  - Number of boosting rounds to be used (higher value usually means more complex model)                               - int (optional)
                learning_rate - How much model weights are adjusted with respect to the loss gradient (determines the step size at each iteration) - float (optional)
                max_depth     - Maximum depth of individual trees                                                                                  - int (optional)
        
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
        
        train - Trains the model using the training dataset
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, threshold = 0.6, n_estimators=100, learning_rate=0.1, max_depth=5):
        super().__init__(train_file, test_file, threshold)
        
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        
        self.model_type = "GB"
        
          
        
class XGBoostModel(SKLM):
    '''
    Class for creating an eXtreme Gradient Boosting ML algorithm - Inherits from Sklearn in the SuperModels file



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
        train_input      - The training data                                                          - list of lists
        test_input       - The testing data                                                           - list of lists
        model_type       - String for filename                                                        - str
        timestamp        - String for uniqueness of filename                                          - str
        filename         - String that the saved file will be named                                   - str
        to_save          - Data to be contained in saved file                                         - dict



    Methods:
        __init__ - Initializes the eXtreme Gradient Boosting model with given parameters and prepares the model for training
        
            Parameters:
                train_file    - Location of dataset file to be used for training                                                               - str
                test_file     - Location of dataset file to be used for testing                                                                - str
                threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)                                     - float (optional)
                n_estimators  - Number of boosting rounds to be used (higher value usually means more complex model)                           - int (optional)
                learning_rate - How much model weights are adjusted with respect to loss gradient - determines the step size at each iteration - float (optional)
                max_depth     - Maximum depth of individual trees                                                                              - int (optional)
        
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
        
        train - Trains the model using the training dataset
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, threshold = 0.55, n_estimators = 100, learning_rate = 0.1, max_depth = 7):
        super().__init__(train_file, test_file, threshold)
    
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,  
            learning_rate=learning_rate,  
            max_depth=max_depth,   
        )
        
        self.model_type = "XGB"
        


class CatBoostModel(SKLM):
    '''
    Class for creating an CatBoost ML algorithm - Inherits from Sklearn in the SuperModels file



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
        train_input      - The training data                                                          - list of lists
        test_input       - The testing data                                                           - list of lists
        model_type       - String for filename                                                        - str
        timestamp        - String for uniqueness of filename                                          - str
        filename         - String that the saved file will be named                                   - str
        to_save          - Data to be contained in saved file                                         - dict



    Methods:
        __init__ - Initializes the Cat Boost model with given parameters and prepares the model for training
        
            Parameters:
                train_file    - Location of dataset file to be used for training                                                               - str
                test_file     - Location of dataset file to be used for testing                                                                - str
                threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)                                     - float (optional)
                iterations    - The number of boosting iterations for the model training                                                       - int (optional)
                learning_rate - How much model weights are adjusted with respect to loss gradient - determines the step size at each iteration - float (optional)
                depth         - The maximum depth of the decision trees used in boosting                                                       - int (optional)   
                l2_leaf_reg   - L2 regularization term for the leaf nodes to prevent overfitting                                               - float (optional)
                subsample     - Fraction of the data to be used for training each tree (controls overfitting)                                  - float (optional)
        
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
        
        train - Trains the model using the training dataset
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, threshold=0.5, iterations=500, learning_rate=0.009, depth=6, 
                 l2_leaf_reg=3, subsample=1):
        
        super().__init__(train_file, test_file, threshold)
        
        self.model = CatBoostClassifier(
            iterations=iterations, 
            learning_rate=learning_rate, 
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            subsample=subsample,
            verbose=0  
        )
        
        self.model_type = 'CatBoost'
        
                 
        
class RandomForestModel(SKLM):
    '''
    Class for creating a Random Forest ML algorithm - Inherits from Sklearn in the SuperModels file



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
        train_input      - The training data                                                          - list of lists
        test_input       - The testing data                                                           - list of lists
        model_type       - String for filename                                                        - str
        timestamp        - String for uniqueness of filename                                          - str
        filename         - String that the saved file will be named                                   - str
        to_save          - Data to be contained in saved file                                         - dict



    Methods:
        __init__ - Initializes the Random Forest model with given parameters and prepares the model for training
        
            Parameters:
                train_file        - Location of dataset file to be used for training                                     - str
                test_file         - Location of dataset file to be used for testing                                      - str
                threshold         - The decision/cutoff boundary (likelihood of fraud that is flagged as such)           - float (optional)
                n_estimators      - Number of boosting rounds to be used (higher value usually means more complex model) - int (optional)
                max_depth         - Maximum depth of individual trees                                                    - int (optional)
                min_samples_split - Minimum number of samples required to split an internal node                         - int (optional)
                min_samples_leaf  - Minimum number of samples required to be at a leaf node                              - int (optional)
        
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
        
        train - Trains the model using the training dataset
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, threshold = 0.48, n_estimators = 1000, max_depth = 9, min_samples_split = 9, min_samples_leaf = 3): 
        super().__init__(train_file, test_file, threshold)

        self.model = RandomForestClassifier(
            oob_score=True,         
            n_jobs=-1,                 
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        self.model_type = "RF"



class SGDClassifierModel(SKLM):
    '''
    Class for creating a Stochastic Gradient Descent algorithm - Inherits from Sklearn in the SuperModels file



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
        train_input      - The training data                                                          - list of lists
        test_input       - The testing data                                                           - list of lists
        model_type       - String for filename                                                        - str
        timestamp        - String for uniqueness of filename                                          - str
        filename         - String that the saved file will be named                                   - str
        to_save          - Data to be contained in saved file                                         - dict



    Methods:
        __init__ - Initializes the Stochastic Gradient Descent model with given parameters and prepares the model for training
        
            Parameters:
                train_file        - Location of dataset file to be used for training                                     - str
                test_file         - Location of dataset file to be used for testing                                      - str
                threshold         - The decision/cutoff boundary (likelihood of fraud that is flagged as such)           - float (optional)
                alpha             - regularization term to prevent overfitting                                           - float (optional)
                max_iter          - Maximum number of iterations for the solver to converge                              - int (optional)
        
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
        
        train - Trains the model using the training dataset
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, threshold=0.5, alpha=1.65, max_iter=100000):
        
        super().__init__(train_file, test_file, threshold)
        
        self.model = SGDClassifier(loss='log_loss', alpha=alpha, max_iter=max_iter)
        
        self.model_type = 'SGD Classifier'


    


                
def main():  # Example code
    train_file = undersample(toList("/Users/connorallan/Desktop/DOJO_project/ML/DataSets/fraudTrain.csv")) # toList only necessary if input is a csv file
    test_file = toList("/Users/connorallan/Desktop/DOJO_project/ML/DataSets/fraudTest.csv")
    
    print("_________________________________________________________________________")
    print("Neural Network")
    model = NeuralNetwork(train_file, test_file)
    model.commenceTraining() 
    #model.saveModel()
    
    print("_________________________________________________________________________")
    print("Logistic Regression")
    model = LogisticRegressionModel(train_file, test_file)
    model.commenceTraining()
    #model.saveModel()
    
    print("_________________________________________________________________________")
    print("Naives Bayes")
    model = NaiveBayes(train_file, test_file)
    model.commenceTraining()
    #model.saveModel()
    
    print("_________________________________________________________________________")
    print("Stochastic Gradient Descent model")
    model = SGDClassifierModel(train_file, test_file)
    model.commenceTraining() 
    #model.saveModel()
    
    print("_________________________________________________________________________")
    print("Gradient Boosting Machine")
    model = GradientBoostingMachineModel(train_file, test_file)
    model.commenceTraining()
    #model.saveModel()
    
    print("_________________________________________________________________________")
    print("XGBoost")
    model = XGBoostModel(train_file, test_file)
    model.commenceTraining()
    #model.saveModel()
    
    print("_________________________________________________________________________")
    print("CatBoost")
    model = CatBoostModel(train_file, test_file)
    model.commenceTraining() 
    #model.saveModel()'''
    
    print("_________________________________________________________________________")
    print("Random Forest")
    model = RandomForestModel(train_file, test_file) 
    model.commenceTraining()
    #model.saveModel()

    
    
    
    
    
if __name__ == "__main__":
    main()