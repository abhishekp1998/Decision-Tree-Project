import numpy as np
import math as mt 

class PriorProbability():
    def __init__(self):
        """
        This is a simple classifier that only uses prior probability to classify 
        points. It just looks at the classes for each data point and always predicts
        the most common class.

        """
        self.most_common_class = None

    def fit(self, features, targets):
        """
        Implement a classifier that works by prior probability. Takes in features
        and targets and fits the features to the targets using prior probability.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N 
                examples.
        """

        Hash = dict() 
        for i in range(len(targets)):
            if(targets[i] in Hash.keys()):
                 Hash[targets[i]] += 1 
            else: 
                Hash[targets[i]] = 1 
    
        Max = -1 
        val = 0 
        for i in Hash:
            if(val < Hash[i]):
                Max = i
                val = Hash[i]
        self.most_common_class = Max




    def predict(self, data):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """
        return np.asarray([self.most_common_class] * len(data))