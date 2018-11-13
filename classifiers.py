from pprint import pprint

import numpy as np

from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    ["Neural Net", MLPClassifier(max_iter=1000)],
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

cv = 5
for set_name, X, y in sets:
    print("-" * 5, set_name, "-" * 5)
    results = {}

    for name, clf in classifiers:
        pipeline = make_pipeline(StandardScaler(), clf)
        result = cross_validate(pipeline, X, y, cv=5, n_jobs=4)
        for key, value in result.items():
            result[key] = np.mean(value).round(decimals=4)
        results[name] = result
    pprint(results)
