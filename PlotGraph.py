import CreateModel as Models
import numpy as np
import matplotlib.pyplot as plt
train_file = "./sparkov/fraudTrain.csv"
test_file = "./sparkov/fraudTest.csv"
#Takes confusion matrix (model.cm) and calculates precision and recall
def metrics(cm):
    [[tn, fp],[fn, tp]] = cm
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

#lower bound, upperbound, islog and n specify the x axis:
#if isLog, then lower bound + upper bound are powers of 10 (ie -5, 0 means 10^-5 to 1)
#else just normal lower and upper bound
#n is the number of values on the x axis
#modelFunc: 
#need c => model(taking hyperparameter c)

#currently plotting precision and recall 
def plotGraph(lowerBound,upperBound, isLog, n, modelFunc, graphTitle, xAxisTitle):
    if(isLog):
        testVals = np.logspace(lowerBound,upperBound,n)
    else:
        testVals = np.arange(lowerBound, upperBound, (upperBound - lowerBound)/n)
    print(testVals)
    precision = [] #true positives/ predicted positives
    recall = [] #true positives/ all positives
    for c in testVals:  
        model = modelFunc(c)
        model.train()
        model.test()
        p, r = metrics(model.cm)
        precision.append(p)
        recall.append(r)
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(testVals, precision, label="Precision", marker="o")
    plt.plot(testVals, recall, label="Recall", marker="s")
    if(isLog):
        plt.xscale("log") 
    plt.xlabel(xAxisTitle)
    plt.ylabel("Score")
    plt.title(graphTitle)
    plt.legend()
    plt.grid(True)
    plt.savefig("./graphs/"+graphTitle+".png") #save image here
    plt.show()

#example use
def genLogisticRegressionGraph():
    plotGraph(-4,-2,True,5,lambda c : Models.LogisticRegressionModel(train_file, test_file, C=c), 
              "Logistic Regression - Regularisation", "Regularisation Parameter")
    plotGraph(0.8,1,False,5, lambda t: Models.LogisticRegressionModel(train_file, test_file,threshold = t),
             "Logistic Regression - Threshold", "Threshold ")

def genNaivesBayes():
    plotGraph(0.8,1,False, lambda t: Models.NaiveBayes(train_file, test_file,threshold = t),
              "NaiveBayes - Threshold", "Threshold ")
    plotGraph(10^-6,10^-12,True, lambda v: Models.NaiveBayes(train_file, test_file,var_smoothing=v),
              "NaiveBayes - varSmoothing", "varSmoothing")

def genNeuralNetwork():
    plotGraph(0.8,1,False, lambda m: Models.NeuralNetwork(train_file, test_file, momentum=m),
              "NeuralNetwork - Momentum", "Momentum")
    plotGraph(0.95,1,False, lambda t: Models.NeuralNetwork(train_file, test_file,threshold = t),
              "NueralNetwork - Threshold", "Threshold ")
    plotGraph(10^-12,10^-2,True, lambda a: Models.NeuralNetwork(train_file, test_file,alpha= a),
              "NueralNetwork - alpha", "alpha")



def main():
    genLogisticRegressionGraph()
    genNaivesBayes()
    genNeuralNetwork()

if __name__ == "__main__":
    main()
