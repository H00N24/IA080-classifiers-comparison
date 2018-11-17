from pprint import pprint

import numpy as np

from data_loader import DataLoader

from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, make_scorer, accuracy_score

# Classifiers
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# from sklearn.gaussian_process import GaussianProcessClassifier

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
    ["QDA", QuadraticDiscriminantAnalysis()],
]

sets = (
    ("iris", *datasets.load_iris(return_X_y=True)),
    ("digits", *datasets.load_digits(return_X_y=True)),
    ("wine", *datasets.load_wine(return_X_y=True)),
    ("cancer", *datasets.load_breast_cancer(return_X_y=True)),
)

data_loader = DataLoader("metal-data/")

cv = 5
for set_name, X, y in data_loader.loaded_data:
    print("-" * 5, set_name, "-" * 5)
    results = {}

    for name, clf in classifiers:
        if isinstance(clf, MLPClassifier):
            pipeline = make_pipeline(StandardScaler(), clf)
        else:
            pipeline = clf
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
        )
        for key, value in result.items():
            result[key] = np.mean(value).round(decimals=4)
        results[name] = result
    pprint(results)
