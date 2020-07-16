import numpy as np
import math 
import copy

class Tree():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)

        """
        self.attribute_names = attribute_names
        self.tree = None
        return

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        """
        self._check_input(features)
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
        self.tree = self.id3(self.attribute_names, features, targets, Max)

    def id3(self, attributes, features, targets, default):
        if np.unique(targets).size == 1:
            return Tree(targets[0])
        elif len(features[0]) == 0:
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
            return Tree(i)
        else:
            best_attribute = self.choose(features, targets, attributes)
            sub_attribute = copy.deepcopy(attributes)
            index = sub_attribute.index(best_attribute)
            sub_attribute.pop(index)
            tree = Tree(None, best_attribute, self.attribute_names.index(best_attribute))
            new_features, new_targets = self.split(features, targets, index)
            for val in range(2):
                Hash = dict() 
                for i in range(len(new_targets[val])):
                    if(new_targets[val][i] in Hash.keys()):
                        Hash[new_targets[val][i]] += 1 
                    else: 
                        Hash[new_targets[val][i]] = 1 
                Max = -1 
                val = 0 
                for i in Hash:
                    if(val < Hash[i]):
                        Max = i
                        val = Hash[i]
                branch = self.id3(sub_attribute, np.asarray(new_features[val]),np.asarray(new_targets[val]), Max)
                tree.branches.append(branch)
            return tree 
    
    
    def split(self, features_i, targets_i, index):
        features = features_i.tolist()
        targets = targets_i.tolist()
        new_features_left = []
        new_features_right = []
        new_targets_left = []
        new_targets_right = []
        for i in range(len(features)):
            if features[i][index] == 0:
                features[i].pop(index)
                new_features_left.append(features[i])
                new_targets_left.append(targets[i])
            else:
                features[i].pop(index)
                new_features_right.append(features[i])
                new_targets_right.append(targets[i])
        return [new_features_left,new_features_right], [new_targets_left, new_targets_right]
    
    
    
    def choose(self, features, targets, attributes):
        high_info_gain = 0
        high_attribute = None
        for i in range(len(attributes)):
            info_gain = information_gain(features, self.attribute_names.index(attributes[i]), targets)
            if info_gain > high_info_gain:
                high_info_gain = info_gain
                high_attribute = attributes[i]
        return high_attribute

    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """
        self._check_input(features)
        predicitions = []
        for feature in features:
            predicitions.append(self.search(feature, self.tree))
        return np.asarray(predicitions)
    
    def search(self, feature, subtree):
        if subtree.attribute_name == "root":
            return subtree.value
        elif feature[subtree.attribute_index] == 0:
            return self.search(feature, subtree.branch[0])
        else:
            return self.search(feature, subtree.branch[1])
    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def information_gain(features_i, attribute_index, targets_i):
    features = features_i.tolist()
    targets = targets_i.tolist()
    set1 = []
    set2 = []
    for x in range(len(features)):
        if(features[x][attribute_index]== 0):
            set1.append(targets[x])
        else:
            set2.append(targets[x])
    entropy = (targets.count(1) / len(features)) * math.log((targets.count(1) / len(features)), 2) + (targets.count(0) / len(features)) * math.log((targets.count(0) / len(features)), 2)
    entropy *= -1
    left_sub_1s = set1.count(1) / len(set1)
    left_sub_0s = set1.count(0) / len(set1)
    right_sub_1s = set2.count(0) / len(set2)
    right_sub_0s = set2.count(1) / len(set2)
    left_entropy = (left_sub_0s * math.log(left_sub_0s, 2) + left_sub_1s * math.log(left_sub_1s, 2)) * -1
    right_entropy = (right_sub_0s * math.log(right_sub_0s, 2) + right_sub_1s * math.log(right_sub_1s, 2)) * -1
    return entropy - (len(set1)/len(features)) * left_entropy - (len(set2)/len(features)) * right_entropy

if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
