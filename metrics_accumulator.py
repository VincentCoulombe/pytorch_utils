import numpy as np
import unittest
from typing import Union, Dict
from sklearn.metrics import mean_absolute_error
from random import randint


class MetricsAccumulator(object):
    def __init__(self) -> None:
        self.running_metrics = {}

    def reset(self):
        self.running_metrics = {}

    def update(
        self,
        batch: Dict[str, Union[float, int]],
    ) -> tuple:

        for name, value in batch.items():
            if isinstance(value, (float, int)):
                self._append_to_running_metrics(name, value)

    def get_metrics_avg(self):
        return {
            name: np.mean(values) if len(values) > 0 else 0.0
            for name, values in self.running_metrics.items()
        }

    def get_scores(self):
        for name in self.running_metrics:
            if "score" in name:
                return self.running_metrics[name]

    def get_losses(self):
        for name in self.running_metrics:
            if "loss" in name:
                return self.running_metrics[name]

    def _append_to_running_metrics(self, name: str, value: Union[float, np.array]):
        if name not in self.running_metrics:
            self.running_metrics[name] = []
        self.running_metrics[name].append(value)


if __name__ == "__main__":

    class TestMetricsAccumulator(unittest.TestCase):
        def setUp(self) -> None:
            self.metrics_accumulator = MetricsAccumulator()

        def test_update(self):
            def run_batch(len_dataset, loss_stack, score_stack):
                running_metrics = self.metrics_accumulator.running_metrics
                for _ in range(len_dataset):
                    loss, score = randint(0, 100), randint(0, 100)
                    self.metrics_accumulator.update({"loss": loss, "score": score})
                    loss_stack.append(loss)
                    score_stack.append(score)

                for name in running_metrics:
                    if name == "loss":
                        self.assertEqual(running_metrics[name], loss_stack)
                        self.assertEqual(
                            self.metrics_accumulator.get_losses(), loss_stack
                        )
                    elif name == "score":
                        self.assertEqual(running_metrics[name], score_stack)
                        self.assertEqual(
                            self.metrics_accumulator.get_scores(), score_stack
                        )

                metrics = self.metrics_accumulator.get_metrics_avg()
                for name in metrics:
                    self.assertGreaterEqual(metrics[name], 0)
                    if name == "loss":
                        self.assertAlmostEqual(
                            metrics[name],
                            sum(loss_stack) / len(loss_stack),
                        )
                        self.assertEqual(running_metrics[name], loss_stack)
                    elif name == "score":
                        self.assertAlmostEqual(
                            metrics[name], sum(score_stack) / len(score_stack)
                        )
                        self.assertEqual(running_metrics[name], score_stack)

            for i in range(1, 20):
                self.metrics_accumulator.reset()
                self.assertDictEqual(self.metrics_accumulator.running_metrics, {})
                len_dataset = randint(1, 100)
                run_batch(len_dataset=len_dataset, loss_stack=[], score_stack=[])
                for name in self.metrics_accumulator.running_metrics:
                    self.assertEqual(
                        len(self.metrics_accumulator.running_metrics[name]), len_dataset
                    )

    unittest.main()
