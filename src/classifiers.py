import numpy as np
import argparse

from data_loader import DataLoader, Saver
from neural_networks import KerasWrapper, create_name

from sklearn import datasets

from sklearn.metrics import make_scorer, accuracy_score
from keras.applications.resnet50 import preprocess_input

# Classifiers
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from keras.applications.resnet50 import ResNet50
from itertools import product

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# Process input arguments
ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", "-d", help="path to metal data")
ap.add_argument("--dataset-str", "-s", help="dataset str")
ap.add_argument("--classifier-str", "-c", help="classifier str")
ap.add_argument("--create-names", "-n", help="create names", action="store_true")
args = vars(ap.parse_args())

if not args["create_names"] and not args["data_dir"]:
    # print("ERROR: No input data")
    exit(-1)

classifiers = [
    ["Nearest Neighbors", KNeighborsClassifier()],
    ["Linear SVM", SVC(kernel="linear", C=0.025)],
    ["RBF SVM", SVC(gamma="scale")],
    ["Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))],
    ["Decision Tree", DecisionTreeClassifier()],
    ["Random Forest", RandomForestClassifier(n_estimators=100)],
    ["Neural Net", MLPClassifier(max_iter=200)],
    ["AdaBoost", AdaBoostClassifier()],
    ["Naive Bayes", GaussianNB()],
    ["QDA", QuadraticDiscriminantAnalysis()],
]

sets = (
    ("iris", *datasets.load_iris(return_X_y=True)),
    ("digits", *datasets.load_digits(return_X_y=True)),
    ("wine", *datasets.load_wine(return_X_y=True)),
    ("cancer", *datasets.load_breast_cancer(return_X_y=True)),
)

layers = ((300, 100), (500, 300), (500, 500, 2000))
act_hidden = ("relu",)
act_output_loss = (
    ("sigmoid", "mean_squared_error"),
    ("softmax", "categorical_crossentropy"),
)
networks = product(layers, act_hidden, act_output_loss)

if args["create_names"]:
    for name, _ in classifiers:
        print(name)
    for network in networks:
        print(create_name(*network))

    exit(0)

metal_data = []
if args["data_dir"] and args["dataset_str"] != "weapon-data":
    # print("Loading metal data...")
    metal_data = DataLoader().get_metal_data(args["data_dir"], args["dataset_str"])
    # print("Loading done")

saver = Saver()

# METAL DATA
for set_name, X, y, y_bin in metal_data:
    # print("-" * 5, set_name, "-" * 5)

    for name, clf in classifiers:
        if args["classifier_str"] and name != args["classifier_str"]:
            continue

        result = cross_validate(
            clf,
            X,
            y,
            cv=5,
            n_jobs=2,
            scoring=({"accuracy": make_scorer(accuracy_score)}),
            error_score=np.nan,
            return_train_score=True,
        )

        results = {}
        for key, value in result.items():
            result[key] = np.mean(value).round(decimals=4)
        results[name] = result

        # Save result
        saver.save_output(results, set_name)

    for params in networks:
        if args["classifier_str"] and create_name(*params) != args["classifier_str"]:
            continue
        results = {}

        model = KerasWrapper(X.shape[1], y_bin.shape[1], *params)
        results = model.cross_validate(X, y, y_bin)

        saver.save_output(results, set_name)

# IMAGE DATA
if args["dataset_str"] != "weapon-data":
    # Load & preprocess data
    image_data, image_labels, y_bin = DataLoader().get_image_data(args["data_dir"])
    image_data = preprocess_input(image_data)

    # print("RetNet50...")
    features = ResNet50(weights="imagenet").predict(image_data)
    # print("Prediction DONE.")
    for name, clf in classifiers:
        if args["classifier_str"] and name != args["classifier_str"]:
            continue

        result = cross_validate(
            clf,
            features,
            image_labels,
            cv=5,
            n_jobs=2,
            scoring=({"accuracy": make_scorer(accuracy_score)}),
            error_score=np.nan,
            return_train_score=True,
        )

        results = {}
        for key, value in result.items():
            result[key] = np.mean(value).round(decimals=4)
        results[name] = result

        # Save result
        saver.save_output(results, "weapon-data")

    for params in networks:
        if args["classifier_str"] and create_name(*params) != args["classifier_str"]:
            continue
        results = {}

        model = KerasWrapper(features.shape[1], y_bin.shape[1], *params)
        results = model.cross_validate(features, image_labels, y_bin)

        saver.save_output(results, "weapon-data")
