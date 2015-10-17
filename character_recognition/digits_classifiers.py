import logging

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__file__)


def get_dataset(images, labels, no_images):
    'returns a tuple with features and targets'
    with open(images, 'rb') as f:
        f.read(16)
        X = np.array(list(f.read(28*28*no_images)), dtype=np.uint8)
        X = X.reshape(no_images, 28*28)

    with open(labels, 'rb') as f:
        f.read(8)
        y = np.array(list(f.read(no_images)))
    return X, y


def get_datasets():
    logger.info("getting datasets")
    X_train, y_train = get_dataset("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000)
    X_val, y_val = get_dataset("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 10000)
    return (X_train, y_train, X_val, y_val)


classifiers = {
    'SVM': SGDClassifier,
    'Logistic regression': lambda: SGDClassifier(loss='log'),
}


def try_different_classifiers(X_train, y_train, X_val, y_val):
    for name in classifiers.keys():
        get_predictor(name, X_train, y_train, X_val, y_val)


def get_predictor(name, X_train=None, y_train=None, X_val=None, y_val=None):
    if X_train is None:
        X_train, y_train, X_val, y_val = get_datasets()
    classifier = classifiers[name]()
    logger.info("Training %s", name)
    classifier.fit(X_train, y_train)

    logger.info("Predicting %s", name)
    predictions = classifier.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    logger.info("Accuracy of %s: %s%%", name, accuracy*100)
    return classifier

if __name__ == '__main__':
    X_train, y_train, X_val, y_val = get_datasets()
    try_different_classifiers(X_train, y_train, X_val, y_val)
