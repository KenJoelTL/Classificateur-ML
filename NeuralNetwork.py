import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

import time
import math
import sys

# Building a Neural Network
class NeuralNetwork():
    # Initialize Neural Network
    def __init__(self, X, Y, label, alpha, nbIter, hiddenNodes):
        self.alpha = float(alpha)
        self.nbIter = int(nbIter)
        self.hiddenNodes = int(hiddenNodes )
        np.set_printoptions(threshold=sys.maxsize)
        print("X shape : " + str(X.shape))
        print("Number of examples : " + str(len(X)))
        print("Number of features per exemple : " + str(len(X[1])))
        print("Number of classes : " + str(Y.shape[1]))
        print("Number of iteration : " + str(self.nbIter))
        print("Alpha : " + str(self.alpha))
        print("Number of hidden nodes : " + str(self.hiddenNodes))

    # Split dataset into 2 datasets (for Validation and Testing)
    @staticmethod
    def validation(X, Y, size):
        # ----------
        #This code is taken from this website :
        #https://scikit-learn.org/stable/modules/cross_validation.html
        XTrain, XTest, YTrain, YTest = train_test_split(X, Y
            , test_size=size, random_state=0)
        # ----------
        return XTrain, XTest, YTrain, YTest

    # Sigmoid
    def activation(self, z):
        f = 1/(1 + np.exp(-z))
        return f

    # Sigmoid derivation
    def derivation(self, z):
        f = self.activation(z)*(1-self.activation(z))
        return f
    
    # Forward propagation
    def propagation(self, X, b1, b2):
        a1 = np.true_divide(X, 255)
        z2 = np.dot(a1, self.w1) + b1
        a2 = self.activation(z2)
        z3 = np.dot(a2, self.w2) + b2
        a3 = self.activation(z3)
        a3 = np.where(a3 >= 0.5, 1, 0)
        return a2, a3, z2, z3, b1, b2

    # Backward propagation
    def retropropagation(self, a2, a3, z2, X, Y, m, saveW, b1, b2):
        # Calculate delta 3
        d3 = a3 - Y
        # Calculate delta 2
        temp=np.dot(d3, self.w2.T)
        d2 = temp*self.derivation(z2)
        # Calculate Big Delta 1 and 2 for weight and bias
        bigDelta1 = (1/m)*np.dot(X.T, d2)
        bigDelta1b1 = (1/m)*np.sum(d2, axis=1, keepdims=True)
        bigDelta2 = (1/m)*np.dot(a2.T, d3)
        bigDelta2b2 = (1/m)*np.sum(d3, axis=1, keepdims=True)

        # If the weight need to be save, reajust the weight
        if (saveW == True):
            self.w1 = self.w1-self.alpha*bigDelta1
            b1 = b1-self.alpha*bigDelta1b1
            self.w2 = self.w2-self.alpha*bigDelta2
            b2 = b2-self.alpha*bigDelta2b2
        return d2, d3, b1, b2

    # Calculate logistic regression cost
    def cost(self, a3, Y,m):
        with np.errstate(divide='ignore'):
        # ----------
        # this code was taken from there : https://datascience.stackexchange.com/questions/22470/python-implementation-of-cost-function-in-logistic-regression-why-dot-multiplic
            cost = -1/m * np.sum( np.dot(np.log(a3+1e-9), Y.T) + np.dot(np.log(1-a3+1e-9), (1-Y.T)))
        #    cost = np.sum(- np.dot(Y, np.log(a3).T) - np.dot(1 - Y, np.log(1 - a3).T))/m
        # -----
        #print("Cost : " + str(cost))
        return cost

    # Change from One Hot Encoding to string labels
    def reverseOneEncoding(self, Y):
        out = "";
        self.label = [ 
                "Cercle2",
                "Cercle3",
                "Diamant2",
                "Diamant3",
                "Hexagone2",
                "Hexagone3",
                "Triangle2",
                "Triangle3",
        ]
        output = np.empty(len(Y),dtype=object)
        # Change each output into a label
        for i, val in enumerate(Y):
            if (val==[0,0,0,0,0,0,0,1]).all():
                out = self.label[0]
            elif (val==[0,0,0,0,0,0,1,0]).all():
                out = self.label[1]            
            elif (val==[0,0,0,0,0,1,0,0]).all():
                out = self.label[2]           
            elif (val==[0,0,0,0,1,0,0,0]).all():
                out = self.label[3]
            elif (val==[0,0,0,1,0,0,0,0]).all():
                out = self.label[4]
            elif (val==[0,0,1,0,0,0,0,0]).all():
                out = self.label[5]
            elif (val==[0,1,0,0,0,0,0,0]).all():
                out = self.label[6]
            elif (val==[1,0,0,0,0,0,0,0]).all():
                out = self.label[7]
            output[i] = out
        return output

    # Show metrics
    def metrics(self, a3, Y):
        # Change from One Hot Encoding to string labels
        Y = self.reverseOneEncoding(Y)
        a3 = self.reverseOneEncoding(a3)

        # Print metrics
        metricsResult = metrics.classification_report(Y, a3, digits=4, zero_division=0)
        print(metricsResult)
        print("----- Confusion matrix -----")
        print("predicted column : " + str(self.label))
        print(confusion_matrix(Y, a3, labels=self.label))
        return metricsResult

    # Show duration of step
    def Duration(self, startTime):
        # --------
        # this code was took from this website :
        # https://stackoverflow.com/questions/3620943/measuring-elapsed-time-with-the-time-module/46544199
        duration = time.time() - startTime
        duration = time.strftime("%H:%M:%S", time.gmtime(duration))
        # --------
        print(duration)

    
    def train(self, X, Y, pourcentage):
        # Create weight
        self.w1 = np.random.randn(X.shape[1], self.hiddenNodes)
        self.w2 = np.random.randn(self.hiddenNodes, 8)
        
        # Create empty errors array
        errorsTrain = [] 
        errorsValidation = [] 

        # Split Train dataset into Train and Validation dataset
        XTrain, XValidate, YTrain, YValidate = self.validation(X, Y, 0.2)

        # If test scaling, cut a part of Train dataset
        if(pourcentage != 1.0):
            XTrain, _, YTrain, _ = self.validation(XTrain, YTrain, 1-float(pourcentage))
            
        # Prepare bias for both Train and Validation
        b1 = np.ones((XTrain.shape[0], self.hiddenNodes))
        b2 = np.ones((XTrain.shape[0], 8))
        b1V = np.ones((XValidate.shape[0], self.hiddenNodes))
        b2V = np.ones((XValidate.shape[0], 8))
        
        print("----- Starting training -----")
        startTime = time.time()

        # Start iterations
        for i in range(self.nbIter):
            # Quick prints of progress
            if (i == math.trunc(self.nbIter/2)):
                print("...Half way there...")
            elif (i == math.trunc(self.nbIter/4)):
                print("...1/4 done...")
            elif (i == math.trunc(self.nbIter*0.75)):
                print("...3/4 done...")

            # Forward propagation
            a2, a3, z2, z3, b1, b2 = self.propagation(XTrain, b1, b2)

            # Backward propagation
            d2, d3, b1, b2 = self.retropropagation(a2, a3, z2, 
                    XTrain, YTrain, XTrain.shape[1], True,b1 ,b2)

            # Calculate cost
            cost = self.cost(a3, YTrain, XTrain.shape[1])
            errorsTrain.append(cost)
            end = False

            # Check if Validation needed each 5 iterations
            if (i % 5 == 0):
                # Forward propagation for Validation
                a2V, a3V, z2V, z3V, b1V, b2V = self.propagation(XValidate, b1V, b2V)

                # Backward propagation for Validation
                d2V, d3V, b1V, b2V = self.retropropagation(a2V, a3V, z2V, 
                        XValidate, YValidate, XValidate.shape[1], False, b1V, b2V)

                # Calculate Validation cost
                costValidation = self.cost(a3V, YValidate, XValidate.shape[1])

                # Saving cost for the graph
                for idx in range(5):
                    errorsValidation.append(costValidation)

                # Early stopping if the loss is greater then the previous value
                if(len(errorsValidation) > 5):
                    if(errorsValidation[-1] > errorsValidation[-6]):
                        end = True
                        print("!!! Early Stopping at iteration " + str(i) + " !!!")
            # End of iteration or if early stopping triggered
            if (i == self.nbIter - 1 or end == True):
                print("----- Ending training -----")
                self.Duration(startTime)
                # Calculate metrics
                self.metrics(a3, YTrain)
                break;
        return self, errorsTrain, errorsValidation

    # Test our Neural Network Dataset
    def test(self, X, Y):
        # Prepare bias
        b1T = np.ones((X.shape[0], self.hiddenNodes))
        b2T = np.ones((X.shape[0], 8))

        # Testing data into propagation and retropropagation
        print("----- Testing new data -----")
        startTime = time.time()
        a2, a3, z2, z3, b1T, b2T = self.propagation(X, b1T, b2T)
        d2, d3, b1T, b2T = self.retropropagation(a2, a3, z2, X, Y, X.shape[1], False, b1T, b2T)
        self.Duration(startTime)

        # Show metrics
        self.metrics(a3, Y)
        return self

    # Compare with other models
    def Compare(self, X, Y, model):
        # Processing X et Y
        X = np.array([image.flatten() for image in X])
        le = preprocessing.LabelEncoder()
        Y = le.fit_transform(Y)
        
        # Seperate Train and Test dataset
        XTrain, XTest, YTrain, YTest = self.validation(X, Y, 0.2)

        # Check which model to do
        if model == "Linear":
            model = LinearRegression()
            print("----- Starting linearRegression training -----")
        elif model == "Bayes":
            model = GaussianNB()
            print("----- Starting Naive Bayes training -----")
        elif model == "KNN":
            model = KNeighborsClassifier(n_neighbors=3)
            print("----- Starting KNN training -----")

        # Start training
        startTime = time.time()
        model.fit(XTrain, YTrain)
        print("----- Ending training -----")
        self.Duration(startTime)
        
        # Start testing
        print("----- Testing new data -----")
        startTime = time.time()
        YPrime = model.predict(XTest)
        self.Duration(startTime)
        
        # Print quick metrics
        yPrime = np.around(YPrime)
        fScore = f1_score(yPrime, YTest, average='weighted')
        accuracy = accuracy_score(yPrime, YTest)
        precision = precision_score(yPrime, YTest, average='weighted')
        recall = recall_score(yPrime, YTest, average='weighted')

        results = [accuracy, precision, recall, fScore]
        
        print("Accuracy: " + "{:.2%}".format(results[0]))
        print("Precision: " + "{:.2%}".format(results[1]))
        print("Recall: " + "{:.2%}".format(results[2]))
        print("F-Score: " + "{:.2%}".format(results[3]) + "\n")


