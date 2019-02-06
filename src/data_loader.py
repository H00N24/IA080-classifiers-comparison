from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    StandardScaler,
    LabelEncoder,
    LabelBinarizer,
)
from keras.preprocessing.image import img_to_array, load_img
from datetime import datetime

import numpy as np

import glob
import json
import os
import re


class DataLoader:
    def get_image_data(
        self, data_dir, width=224, height=224, correct_dataset_size=True
    ):
        # print("Loading image data...")
        labels_dict = dict(
            (category, i) for i, category in enumerate(os.listdir(data_dir))
        )

        cat_counter = []
        image_paths = []
        for cat_path in glob.glob(data_dir + "/*"):
            cat_list = [
                cat_path + "/" + image_name for image_name in os.listdir(cat_path)
            ]
            image_paths += cat_list
            if correct_dataset_size:
                cat_counter.append(len(cat_list))

        # Correct dataset size
        max_images = min(cat_counter) if correct_dataset_size else 0

        num_of_images_per_category = dict((label, 0) for label in labels_dict)

        image_labels = []
        image_data = []
        for path in image_paths:
            label = path.split(os.path.sep)[-2]

            # Correct dataset size
            if correct_dataset_size and num_of_images_per_category[label] >= max_images:
                continue

            try:
                image = load_img(path, target_size=(width, height))
                image = img_to_array(image)
                image_data.append(image)
            except IOError:
                # print("Invalid image: " + path)
                continue
            # image_labels.append(labels_dict[label])
            image_labels.append(label)

            if correct_dataset_size:
                num_of_images_per_category[label] += 1

        image_data = np.array(image_data, dtype="float16")
        # image_labels = np.array(image_labels)

        assert len(image_data) == len(image_labels)

        # print("Loaded {0} images.".format(len(image_data)))

        lab_enc = LabelEncoder()
        lab_bin = LabelBinarizer()

        y = lab_enc.fit_transform(image_labels)
        y_bin = lab_bin.fit_transform(image_labels)

        return (image_data, y, y_bin)

    def get_metal_data(self, data_dir, dataset_str, normalize_numerical=False):
        self._init_metal_data(data_dir, dataset_str, normalize_numerical)

        for data_file_path, names_file_path in zip(self.data_files, self.names_files):
            encoder, numeric = self._load_names_file(names_file_path)
            X, y, n_classes = self._load_data_file(data_file_path, encoder, numeric)
            yield (data_file_path, X, y, n_classes)

    def _init_metal_data(self, data_dir, dataset_str, normalize_numerical):
        self.data_dir = data_dir

        files = [x for x in os.listdir(self.data_dir) if dataset_str in x]
        self.data_files = sorted([x for x in files if x.endswith(".data")])
        self.names_files = sorted([x for x in files if x.endswith(".names")])

        num_steps = [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="median"))
        ]
        if normalize_numerical:
            num_steps.append(("scaler", StandardScaler()))
        self.numerical_transformer = Pipeline(steps=num_steps)
        self.categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OrdinalEncoder()),
            ]
        )

    def _load_names_file(self, names_file_path: str) -> tuple:
        """Read categories and creates pipeline for data transformation

        Args:
            names_file_path (str): names_file path

        Returns:
            tuple: ColumnTransformer, array with True on positions of continuous values
        """

        with open(os.path.join(self.data_dir, names_file_path)) as names_file:
            numeric = []
            transformers = []
            column_index = 0
            for index, line in enumerate(names_file):
                line = line.strip().lower()
                if not line:
                    continue
                if index == 0:
                    continue

                res = re.search(r"^(?P<name>.*):\s*(?P<types>.*)\.", line)
                if not res:
                    # print("Error:", names_file_path, file=sys.stderr)
                    # print("Error:", line, file=sys.stderr)
                    continue

                types = res.group("types").strip()

                if types == "continuous":
                    numeric.append(True)
                else:
                    transformers.append(
                        (
                            "line {}".format(column_index),
                            self.categorical_transformer,
                            [column_index],
                        )
                    )
                    numeric.append(False)
                column_index += 1
            """
            transformers.append(
                (
                    "line ".format(column_index),
                    self.categorical_transformer,
                    [column_index],
                )
            )"""
            numeric.append(False)

            encoder = ColumnTransformer(
                transformers=transformers, remainder=self.numerical_transformer
            )

            return (encoder, numeric)

    def _load_data_file(
        self, data_file_path: str, encoder: ColumnTransformer, numeric: list
    ) -> tuple:
        with open(os.path.join(self.data_dir, data_file_path)) as data_file:
            data = []
            labels = []
            for line in data_file:
                line = line.strip().lower()
                if not line:
                    continue
                fields = np.array(line.split(","))
                fields[numeric] = np.genfromtxt(fields[numeric])
                data.append(fields[:-1])
                labels.append(fields[-1])

                # data.append(fields)

            transformed_data = encoder.fit_transform(data)
            X = transformed_data[:, :-1]

            lab_enc = LabelEncoder()
            lab_bin = LabelBinarizer()

            y = lab_enc.fit_transform(labels)
            y_bin = lab_bin.fit_transform(labels)
            # y = transformed_data[:, -1:].flatten()

            return (X, y, y_bin)


class Saver:
    def __init__(self):
        self.datetime_now = datetime.now()

    def save_output(self, results, set_name):
        for key, item in results.items():
            metrics = ("fit_time", "test_accuracy", "train_accuracy")
            print(
                set_name[:-5],
                key,
                ";".join([str(item[x]) for x in metrics]),
                sep=";",
                flush=True,
            )
            #exit(0)
        # pprint(results)

    def _save_output(self, results, set_name):
        path = "output/"
        if not os.path.exists(path):
            os.mkdir(path)

        path += self.datetime_now.strftime("%Y-%m-%d-%H:%M:%S") + "/"
        if not os.path.exists(path):
            os.mkdir(path)

        path += set_name.replace(".", "-") + ".json"
        write_option = "w" if not os.path.exists(path) else "a"

        json_file = open(path, write_option)
        json_file.write(
            json.dumps(results, sort_keys=False, indent=4, separators=(",", ":"))
        )
        json_file.close()

        # print("-- Result saved! --")
