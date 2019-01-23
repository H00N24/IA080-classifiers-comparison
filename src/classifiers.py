from pprint import pprint

import numpy as np
import argparse

from data_loader import DataLoader, Saver
from neural_networks import KerasWrapper

from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, make_scorer, accuracy_score
from keras.applications.resnet50 import preprocess_input

# Classifiers
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from keras.applications.resnet50 import ResNet50

# from sklearn.gaussian_process import GaussianProcessClassifier

# Process input arguments
ap = argparse.ArgumentParser()
ap.add_argument("--metal", "-m", help="path to metal data")
ap.add_argument("--image", "-i", help="path to image data")
args = vars(ap.parse_args())

if not args['metal'] and not args['image']:
    print("ERROR: No input data")
    exit(-1)

classifiers = [
    ["Nearest Neighbors", KNeighborsClassifier()],
    ["Linear SVM", SVC(kernel="linear", C=0.025)],
    ["RBF SVM", SVC(gamma="scale")],
    # ["Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))],
    ["Decision Tree", DecisionTreeClassifier()],
    ["Random Forest", RandomForestClassifier(n_estimators=100)],
    ["Neural Net", MLPClassifier(max_iter=200)],
    ["AdaBoost", AdaBoostClassifier()],
    ["Naive Bayes", GaussianNB()],
    ["QDA", QuadraticDiscriminantAnalysis()]
]

sets = (
    ("iris", *datasets.load_iris(return_X_y=True)),
    ("digits", *datasets.load_digits(return_X_y=True)),
    ("wine", *datasets.load_wine(return_X_y=True)),
    ("cancer", *datasets.load_breast_cancer(return_X_y=True)),
)

networks = [
    {'activation' : 'tanh'},
    {'activation' : 'sigmoid'},
    {'activation' : 'relu'}
]

metal_data = []
if args['metal']:
    print("Loading metal data...")
    metal_data = DataLoader().get_metal_data(args['metal'])
    print("Loading done")

saver = Saver()

# METAL DATA
for set_name, X, y, y_bin in metal_data:
    print("-" * 5, set_name, "-" * 5)

    for name, clf in classifiers:
        print('[{0}]'.format(name))
        try:
            result = cross_validate(
                clf,
                X,
                y,
                cv=5,
                n_jobs=4,
                scoring=(
                    {
                        "accuracy": make_scorer(accuracy_score),
                        "micro_precision": make_scorer(precision_score, average="micro"),
                        "macro_precision": make_scorer(precision_score, average="macro"),
                    }
                ),
                error_score=np.nan,
                #verbose=True,
            )
        except KeyboardInterrupt:
            print("INTERRUPTED")
            continue

        results = {}
        for key, value in result.items():
            result[key] = np.mean(value).round(decimals=4)
        results[name] = result

        # Save result
        saver.save_output(results, set_name)

    for params in networks:

        model = KerasWrapper(X.shape[1], y_bin.shape[1], **params)

        try:

            skf = StratifiedKFold(n_splits=5, shuffle=True)

            for (train_index, test_index) in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y_bin[train_index], y_bin[test_index]

                model.fit(X_train, y_train)
                print('Done')
                print(model.evaluate(X_test, y_test))

        except KeyboardInterrupt:
            print("INTERRUPTED")
            continue
        

# IMAGE DATA
if args['image']:
    # Load & preprocess data
    image_data, image_labels, labels_dict = DataLoader().get_image_data(args['image'])
    image_data = preprocess_input(image_data)

    print("RetNet50...")
    features = ResNet50(weights='imagenet').predict(image_data)
    print("Prediction DONE.")
    for name, clf in classifiers:
        print('[{0}]'.format(name))
        try:
            result = cross_validate(
                clf,
                features,
                image_labels,
                cv=5,
                n_jobs=4,
                scoring=(
                    {
                        "accuracy": make_scorer(accuracy_score),
                        "micro_precision": make_scorer(precision_score, average="micro"),
                        "macro_precision": make_scorer(precision_score, average="macro"),
                    }
                ),
                error_score=np.nan,
                #verbose=True,
            )
        except KeyboardInterrupt:
            print("INTERRUPTED")
            continue

        results = {}
        for key, value in result.items():
            result[key] = np.mean(value).round(decimals=4)
        results[name] = result

        # Save result
        saver.save_output(results, 'weapon-data')
