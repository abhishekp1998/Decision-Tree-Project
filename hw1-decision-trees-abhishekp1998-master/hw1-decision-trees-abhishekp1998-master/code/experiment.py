from .decision_tree import DecisionTree
from .prior_probability import PriorProbability
from .metrics import precision_and_recall, confusion_matrix, f1_measure, accuracy
from .data import load_data, train_test_split

def run(data_path, learner_type, fraction):
    """
    This function walks through an entire machine learning workflow as follows:

        1. takes in a path to a dataset
        2. loads it into a numpy array with `load_data`
        3. instantiates the class used for learning from the data using learner_type (e.g
           learner_type is 'decision_tree', 'prior_probability')
        4. splits the data into training and testing with `train_test_split` and `fraction`.
        5. trains a learner using the training split with `fit`
        6. tests the trained learner using the testing split with `predict`
        7. evaluates the trained learner with precision_and_recall, confusion_matrix, and
           f1_measure

    Each run of this function constitutes a trial. Your learner should be pretty
    robust across multiple runs, as long as `fraction` is sufficiently high. See how
    unstable your learner gets when less and less data is used for training by
    playing around with `fraction`.

    IMPORTANT:
    If fraction == 1.0, then your training and testing sets should be exactly the
    same. This is so that the test cases are deterministic. The test case checks if you
    are fitting the training data correctly, rather than checking for generalization to
    a testing set.

    Args:
        data_path (str): path to csv file containing the data
        learner_type (str): either 'decision_tree' or 'prior_probability'. For each of these,
            the associated learner is instantiated and used for the experiment.
        fraction (float between 0.0 and 1.0): fraction of examples to be drawn for training

    Returns:
        confusion_matrix (np.array): Confusion matrix of learner on testing examples
        accuracy (np.float): Accuracy on testing examples using learner
        precision (np.float): Precision on testing examples using learner
        recall (np.float): Recall on testing examples using learner
        f1_measure (np.float): F1 Measure on testing examples using learner
    """
    features, targets, attributes = load_data(data_path)
    train_features, train_targets, test_features, test_targets = train_test_split(features, targets, fraction)
    cf = None 
    acc = None 
    prec = None 
    rec = None 
    f1 = None 


    if(learner_type == "prior_probability"):
        classifier = PriorProbability()
        classifier.fit(train_features, train_targets)
        results = classifier.predict(test_features)
        cf = confusion_matrix(test_targets, results)
        acc = accuracy(test_targets, results)
        prec, rec = precision_and_recall(test_targets, results)
        f1 = f1_measure(test_targets, results)
    else:
        classifier = DecisionTree(attributes)
        classifier.fit(train_features, train_targets)
        results = classifier.predict(test_features)
        cf = confusion_matrix(test_targets, results)
        acc = accuracy(test_targets, results)
        prec, rec = precision_and_recall(test_targets, results)
        f1 = f1_measure(test_targets, results)







    # Order of these returns must be maintained
    return cf, acc, prec, rec, f1
