import seaborn as sns
import matplotlib.pyplot as plt






def plotCM(cms, titles):
    '''
    Function for plotting one or more confusion matrices
    
    

    Parameters:
    
        cms    - Array of confusion matrices to be displayed            - list of lists
        titles - Array of titles corresponding to each confusion matrix - list of str              
    '''
        
    num_matrices = len(cms)
    
    # Create grid layout
    rows = num_matrices // 2 + (num_matrices % 2)  
    cols = 3
    
    _, axes = plt.subplots(rows, cols, figsize=(9, 3 * rows))  # Adjust the size based on number of rows
    axes = axes.flatten()

    
    for i, cm in enumerate(cms):
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=axes[i], cbar=False, 
                    xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
        axes[i].set_title(f'{titles[i]} {i + 1}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
        
        
        
def plotP(cm):
    '''
    Function for plotting the predictions of a model on the fraud row of a confusion matrix
    
    
    
    Parameters:
        cm - The confusion matrix used to extract fraud row data - 2D array         
    '''

    fraud_row = cm[0, :]  # This gives you the row corresponding to fraud (1)

    plt.figure(figsize=(6, 2))  # Adjust the figure size
    sns.heatmap([fraud_row], annot=True, fmt='d', cmap="Blues", cbar=False,
                xticklabels=["Not Fraud", "Fraud"], yticklabels=[""])
    plt.title("Prediction Results")
    plt.xlabel('True Label')
    plt.show()
    
    plt.show()