from ImageGenerator import ImageGenerator
from NeuralNetwork import NeuralNetwork
import sys
import matplotlib.pyplot as plt

# Initialize dataset
ig = ImageGenerator()

if len(sys.argv) > 1:
    # Create dataset of images (existing and auto-generated) of 30x30 pixels
    # Command format : python Main.py Generate
    if sys.argv[1] == "Generate":
        print("----- Generating Images -----")
        ig.createDatabase("images_dataset")
    # Test scaling
    # Command format : python Main.py Scale [scale] [alpha] [number of iterations] [number of hidden nodes]
    elif sys.argv[1] == "Scale": 
        # Init the perceptron
        nn = NeuralNetwork(ig.getDatabaseX(),ig.oneHotEncoding(),ig.getDatabaseY(), sys.argv[3], sys.argv[4], sys.argv[5])
        XTrain, XTest, YTrain, YTest = nn.validation(ig.getDatabaseX(), ig.oneHotEncoding(), 0.2)
        # Training with X and Y
        nn, errorsTrain, errorsValidation =  nn.train(XTrain, YTrain, sys.argv[2])

        # Testing with X and Y
        nn =  nn.test(XTest, YTest)
    # Comparing with other models
    # Command format : python Main.py [model]
    elif sys.argv[1] == "Linear" or sys.argv[1] == "Bayes" or sys.argv[1] == "KNN":
        # Init the perceptron while ignoring the neural network hyperparameters
        nn = NeuralNetwork(ig.getDatabaseX(),ig.oneHotEncoding(),ig.getDatabaseY(), 0, 0, 0)


        # Training with X and Y for the specific model
        nn.Compare(ig.getDatabaseX(), ig.getDatabaseY(), sys.argv[1])
    # Training and testing the Neural Network
    # Command format : python Main.py [alpha] [number of iterations] [number of hidden nodes]
    else:
        # Init the perceptron
        nn = NeuralNetwork(ig.getDatabaseX(),ig.oneHotEncoding(),ig.getDatabaseY(), sys.argv[1], sys.argv[2], sys.argv[3])

        XTrain, XTest, YTrain, YTest = nn.validation(ig.getDatabaseX(), ig.oneHotEncoding(), 0.2)

        # Training with X and Y
        nn, errorsTrain, errorsValidation =  nn.train(XTrain,YTrain, 1.0)

        # Testing with X and Y
        nn =  nn.test(XTest,YTest)

        # Plotting
        plt.plot(range(1, len(errorsTrain) + 1), errorsTrain, 'r')
        plt.plot(range(1, len(errorsValidation) + 1), errorsValidation, 'b')
        plt.xlabel('Number of iterations')
        plt.ylabel('Errors')
 
        plt.tight_layout()
        plt.show()
