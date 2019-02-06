import subprocess
import os
from itertools import product, chain


# 10 minutes
TIMEOUT = 10 * 60

classifier_names = (
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "NN-(300, 100)-relu-sigmoid-mean_squared_error",
    "NN-(300, 100)-relu-softmax-crossentropy",
    "NN-(500, 300)-relu-sigmoid-mean_squared_error",
    "NN-(500, 300)-relu-softmax-crossentropy",
    "NN-(500, 500, 2000)-relu-sigmoid-mean_squared_error",
    "NN-(500, 500, 2000)-relu-softmax-crossentropy",
)

metal_dir = "../data/metal-data/"
metal_datasets = sorted([x[:-5] for x in os.listdir(metal_dir) if x.endswith(".data")])

extra_dir = "../data/extra-data"
extra_datasets = sorted([x[:-5] for x in os.listdir(extra_dir) if x.endswith(".data")])

weapon_dir = "../data/weapon-data"
weapon_datasets = ["weapon-data"]

datasets = chain(
    *(
        product((x[0],), x[1])
        for x in (
            (metal_dir, metal_datasets),
            (extra_dir, extra_datasets),
            (weapon_dir, weapon_datasets),
        )
    )
)

print("Dataset;Classifier;Fit time;Test accuracy;Train accuracy")

for dataset_data, name in product(datasets, classifier_names):
    command = [
        "python",
        "classifiers.py",
        "--data-dir",
        dataset_data[0],
        "--classifier-str",
        name,
        "--dataset-str",
        dataset_data[1],
    ]
    try:
        subprocess.call(command, timeout=TIMEOUT, stderr=None)
    except KeyboardInterrupt:
        exit(0)
    except Exception as e:
        e
        print(dataset_data[1], name, TIMEOUT, 0, 0, sep=";", flush=True)
        continue

