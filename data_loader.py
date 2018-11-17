import os
import re
import sys

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder


class DataLoader:
    def __init__(self, data_dir: str, normalize_numerical: bool = False):
        self.data_dir = data_dir

        files = os.listdir(self.data_dir)
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

    @property
    def loaded_data(self) -> tuple:
        """Data for classification

        Returns:
            tuple: (Data, Categories)
        """

        for data_file_path, names_file_path in zip(self.data_files, self.names_files):
            encoder, numeric = self._load_names_file(names_file_path)
            X, y = self._load_data_file(data_file_path, encoder, numeric)
            yield (data_file_path, X, y)

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
                    print("Error:", names_file_path, file=sys.stderr)
                    print("Error:", line, file=sys.stderr)
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
            y = lab_enc.fit_transform(labels)
            # y = transformed_data[:, -1:].flatten()

            return (X, y)
