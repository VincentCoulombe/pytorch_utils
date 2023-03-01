import numpy as np
import unittest


class ScoringFunction(object):
    def __init__(self, func_name: str):
        self.func_name = func_name.lower()

    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        if len(predictions) != len(labels):
            raise ValueError(
                f"Les predictions et les labels n'ont pas la même taille. {predictions} != {labels}"
            )
        return getattr(self, self.func_name)(predictions, labels)

    def mae(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        return np.mean(np.abs(predictions - labels))


if __name__ == "__main__":

    class TestScoringFunction(unittest.TestCase):
        def setUp(self) -> None:
            self.predictions = np.array([1, 2, 3])
            self.labels = np.array([4, 5, 6])

        def test_mae(self):
            scoring_function = ScoringFunction(func_name="mae")
            mae = scoring_function(self.predictions, self.labels)
            self.assertEqual(mae, 3.0)

    unittest.main()
