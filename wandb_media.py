import wandb
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import unittest
import re
from typing import Optional, List, Union, Dict, Iterable


class WandbMedia(object):
    def __init__(self, title: str) -> None:
        """self.data = {nom de la variable: [valeur1, valeur2, ...]}"""
        self.title = title
        self.reset()

    def reset(self):
        self.data = {}

    def add_batch(self, batch: Dict[str, np.ndarray]):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError


class WandbTable(WandbMedia):
    """Une table wandb."""

    def __init__(self, title: str) -> None:
        super().__init__(title)

    def add_batch(self, batch: Dict[str, Iterable]) -> None:
        """Ajoute une batch de données à la table. Si la valeur est un numpy array,
            prend pour acquis que c'est une image.

        Args:
            batch: {nom_de_la_colonne = [valeur1, valeur2, ...]}

        Raises:
            ValueError: Les arrays de la batch doivent avoir toutes les mêmes longueurs.
        """

        batch_length = 0
        for var_name, batch_value in batch.items():
            if isinstance(batch_value, Iterable):
                batch_size = len(batch_value)
                if batch_length == 0:
                    batch_length = batch_size
                elif batch_length != batch_size:
                    raise ValueError(
                        f"Les éléments {list(batch.keys())} n'ont pas toutes la même batch size."
                    )
                for value in batch_value:
                    self._add_to_data(value, var_name)

    def log(self):
        columns, data = self._format_data_to_table()
        wandb.log({self.title: wandb.Table(columns=columns, data=data)})

    def _add_to_data(self, value, col_name):
        if col_name not in self.data:
            self.data[col_name] = []
        if isinstance(value, np.ndarray):
            try:
                self.data[col_name].append(self._numpy_to_image(value))
            except cv2.error as e:
                raise ValueError(
                    f"La table convertis les np.ndarray en image. Hors, {value} est un np.ndarray, mais pas une image."
                ) from e
        elif isinstance(value, (str, int, float, np.integer, np.floating)):
            self.data[col_name].append(value)
        else:
            self.data[col_name].extend(value)

    def _format_data_to_table(self):
        filled_columns = []
        filled_columns_names = []
        nb_rows = 0
        for col_name, column in self.data.items():
            if len(column) > 0:
                if nb_rows == 0:
                    nb_rows = len(column)
                elif nb_rows != len(column):
                    raise ValueError(
                        f"Les colonnes de la table {self.title} ne sont pas toutes de la même longueur."
                    )
                filled_columns.append(column)
                filled_columns_names.append(col_name)

        if len(filled_columns) < 2:
            raise ValueError(
                f"Wandb ne log pas les tables avec moins de 2 colonnes. Les colonnes de la table {self.title} sont {self.title}"
            )

        return filled_columns_names, np.column_stack(tuple(filled_columns)).tolist()

    @staticmethod
    def _numpy_to_image(np_array: np.ndarray) -> wandb.Image:
        """Convertis un np.array en image wandb.

        Args:
            np_array (np.ndarray): Un np.array de la forme (C, H, W)

        Raises:
            ValueError: Le np.array doit être de la forme (C, H, W)

        Returns:
            wandb.Image: Une image wandb
        """
        if len(np_array.shape) != 3:
            raise ValueError(
                f"L'image doit être de la forme (C, H, W). Hors, elle est de la forme {np_array.shape}."
            )
        return wandb.Image(cv2.cvtColor(np_array.transpose(1, 2, 0), cv2.COLOR_BGR2RGB))


class WandbConfusionMatrix(WandbMedia):
    def __init__(self, title: str, classes: List[str]) -> None:
        super().__init__(title)
        self.classes = classes

    def add_batch(self, batch: Dict[str, np.ndarray]):
        """Ajoute une batch de données à la confusion matrix

        Args:
            batch : {"preds": [prediction1, prediction2, ...], "y": [target1, target2, ...]}
        """
        # TODO : à rendre plus générique...
        for variable in batch:
            if variable in self.data:
                self.data[variable].extend(batch[variable].tolist())

    def log(self, label_var_name: str = "y", pred_var_name: str = "preds"):
        cm = confusion_matrix(
            self.data[label_var_name], self.data[pred_var_name], labels=self.classes
        )
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="g",
            cmap="Blues",
            cbar=True,
            xticklabels=self.classes,
            yticklabels=self.classes,
        )
        plt.ylabel("Truth")
        plt.xlabel("Predictions")
        plt.title(self.title)
        wandb.log({self.title: wandb.Image(plt)})


if __name__ == "__main__":

    class WandbMediaTester(unittest.TestCase):
        def test_table(self):
            table = WandbTable("test")
            table.add_batch({"col1": [1, 2, 3], "col2": [4, 5, 6]})
            table.add_batch({"col1": [7, 8, 9], "col2": [3, 2, 1]})
            table.add_batch({"col2": [9, 8, 7], "col1": [7, 9, 8]})
            self.assertEqual(
                table._format_data_to_table(),
                (
                    ["col1", "col2"],
                    [
                        [1, 4],
                        [2, 5],
                        [3, 6],
                        [7, 3],
                        [8, 2],
                        [9, 1],
                        [7, 9],
                        [9, 8],
                        [8, 7],
                    ],
                ),
            )
            self.assertEqual(
                table.data,
                {
                    "col1": [1, 2, 3, 7, 8, 9, 7, 9, 8],
                    "col2": [4, 5, 6, 3, 2, 1, 9, 8, 7],
                },
            )
            self.assertEqual(table.title, "test")

            table.add_batch({"col1": [1, 2, 3]})

            with self.assertRaises(ValueError):
                table._format_data_to_table()
                table.add_batch({"col3": [1, np.array(2), 3], "col4": [4, 5, 6]})
                table.add_batch({"col3": [7, 8], "col4": [3, 2, 1]})
                table.add_batch({"col3": [7, 8, 9], "col4": [3, 2]})

            table.reset()
            self.assertEqual(table.data, {})
            table.add_batch({"col1": [1, 2, 3, 4, 5, 6]})
            with self.assertRaises(ValueError):
                table._format_data_to_table()

    unittest.main()
