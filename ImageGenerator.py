from PIL import Image
import numpy as np
import pickle
from pathlib import Path
import cv2
import random
import Augmentor

# Create a dataset to be used in a Neural Network


class ImageGenerator:
    # Initializing the dataset with images of 30x30 pixels
    def __init__(self):
        self.WIDTH = 30
        self.HEIGHT = 30
        self.LABELS = [
            "Cercle2",
            "Cercle3",
            "Diamant2",
            "Diamant3",
            "Hexagone2",
            "Hexagone3",
            "Triangle2",
            "Triangle3",
        ]

    # Generate randomized images based on the existing ones
    @staticmethod
    def generateImage(folderSource, folderOutput):
        p = Augmentor.Pipeline(folderSource, folderOutput)
        p.rotate90(probability=0.5)
        p.rotate270(probability=0.5)
        p.flip_left_right(probability=0.75)
        p.flip_top_bottom(probability=0.75)
        p.skew_tilt(probability=0.75, magnitude=0.35)
        p.random_distortion(probability=1, grid_width=4,
                            grid_height=4, magnitude=8)
        p.sample(100)

    # Create the dataset
    def createDatabase(self, imageFolder):
        dataDir = Path.cwd() / imageFolder
        dataSet = []

        # Look for each pictures in each folders
        for firstLevel in dataDir.glob("*"):
            if firstLevel.is_dir():
                for secondLevel in firstLevel.glob("*"):
                    if secondLevel.is_dir() and "." not in secondLevel.name:
                        label = secondLevel.name
                        for idx, lab in enumerate(self.LABELS):
                            if label in lab:
                                classIdx = idx
                                ImageGenerator.generateImage(secondLevel, ".")
                                for img in secondLevel.glob("*.jpg"):
                                    # Read the pictures
                                    img = cv2.imread(str(img),
                                                     cv2.IMREAD_GRAYSCALE)
                                    # Resize the pictures to 30x30
                                    resized = cv2.resize(img, (self.WIDTH,
                                                               self.HEIGHT))
                                    # Save the pictures to array
                                    dataSet.append(
                                        [resized, classIdx, label])

        # Shuffles the images
        random.shuffle(dataSet)

        self.X = []
        self.Y = []

        # Taking features and labels from dataset
        for features, class_num, label in dataSet:
            self.X.append(features)
            self.Y.append(label)

        # Converts each image matrix to an image vector
        self.X = np.array(self.X).reshape(-1, self.HEIGHT * self.WIDTH)

        # Creating the files containing all the information about the dataset
        pickle_out = open("X.pkl", "wb")
        pickle.dump(self.X, pickle_out)
        pickle_out.close()

        pickle_out = open("Y.pkl", "wb")
        pickle.dump(self.Y, pickle_out)
        pickle_out.close()

    # Opening an image
    def openImage(self, number):
        # Reshaping image from vector to matrix
        image = self.X[number].reshape(self.HEIGHT, self.WIDTH)
        plt.imshow(image, cmap='gray')
        plt.show()

    # Get the pictures
    def getDatabaseX(self):
        with open("X.pkl", "rb") as db:
            self.X = pickle.load(db)
        return self.X

    # Get the String labels
    def getDatabaseY(self):
        with open("Y.pkl", "rb") as db:
            self.Y = pickle.load(db)
        return self.Y

    # Get the One Hot Encoding version of the String labels
    def oneHotEncoding(self):
        with open("Y.pkl", "rb") as db:
            self.Y = pickle.load(db)
        label = [
            "Cercle2",
            "Cercle3",
            "Diamant2",
            "Diamant3",
            "Hexagone2",
            "Hexagone3",
            "Triangle2",
            "Triangle3",
        ]
        output = np.zeros(shape=(len(self.Y), 8))
        # Change each labels to One Hot Encoding
        for i, val in enumerate(self.Y):
            if val == label[0]:
                out = [0, 0, 0, 0, 0, 0, 0, 1]
            elif val == label[1]:
                out = [0, 0, 0, 0, 0, 0, 1, 0]
            elif val == label[2]:
                out = [0, 0, 0, 0, 0, 1, 0, 0]
            elif val == label[3]:
                out = [0, 0, 0, 0, 1, 0, 0, 0]
            elif val == label[4]:
                out = [0, 0, 0, 1, 0, 0, 0, 0]
            elif val == label[5]:
                out = [0, 0, 1, 0, 0, 0, 0, 0]
            elif val == label[6]:
                out = [0, 1, 0, 0, 0, 0, 0, 0]
            elif val == label[7]:
                out = [1, 0, 0, 0, 0, 0, 0, 0]
            output[i] = out
        return output
