import numpy as np
import unittest


class PredictionsGetter(object):
    def __init__(self, task_type: str):
        self.task_type = task_type.lower()

    def __call__(self, logits: np.ndarray):
        return getattr(self, self.task_type)(logits)

    def regression(self, logits: np.ndarray):
        return logits

    def classification(self, logits: np.ndarray):
        return np.argmax(logits, axis=1)


if __name__ == "__main__":

    class TestPredictionsGetter(unittest.TestCase):
        def setUp(self) -> None:
            self.logits = np.array([[1, 2, 3], [4, 5, 6]])

        def test_regression(self):
            predictions_getter = PredictionsGetter(task_type="regression")
            predictions = predictions_getter(self.logits)
            self.assertEqual(predictions.tolist(), [[1, 2, 3], [4, 5, 6]])
            predictions_getter = PredictionsGetter(task_type="Regression")
            predictions2 = predictions_getter(self.logits)
            self.assertEqual(predictions2.tolist(), predictions.tolist())

        def test_classification(self):
            predictions_getter = PredictionsGetter(task_type="classification")
            predictions = predictions_getter(self.logits)
            self.assertEqual(predictions.tolist(), [2, 2])

    unittest.main()
