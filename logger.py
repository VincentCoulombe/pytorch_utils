import wandb
from typing import Optional, List, Union, Dict, Any
import unittest
import numpy as np
from random import randint

from wandb_media import WandbMedia, WandbTable
from metrics_accumulator import MetricsAccumulator


class WandbRunLogger(object):
    def __init__(
        self,
        wandb_run: wandb,
        prefix: Optional[str] = None,
        sweeping_metric: Optional[str] = None,
        medias: Optional[List[WandbMedia]] = None,
    ) -> None:
        self.wandb_run = wandb_run
        self.medias = [] if medias is None else medias
        self.accumulator = MetricsAccumulator()
        self.sweeping_metric = sweeping_metric
        self.prefix = prefix

    def set_prefix(self, prefix: str):
        self.prefix = prefix

    def set_sweeping_metric(self, sweeping_metric: str):
        self.sweeping_metric = sweeping_metric

    def reset(self):
        self._reset_medias()
        self.accumulator.reset()

    def update(
        self,
        x: np.ndarray,
        preds: np.ndarray,
        y: np.ndarray,
        loss: float,
        score: float,
        **kwargs,
    ):

        args = {
            "x": x,
            "preds": preds,
            "y": y,
            "loss": loss,
            "score": score,
            **kwargs,
        }
        new_args = {}
        for name, value in args.items():
            if name in ["x", "preds", "y"] and "numpy" not in str(type(value)):
                raise ValueError(
                    f"Argument: {name} doit être un numpy array. Est: {type(value)}"
                )
            elif name in ["loss", "score"] and not isinstance(value, (float, int)):
                raise ValueError(
                    f"Argument: {name} doit être un float ou un int. Est: {type(value)}"
                )
            if self.prefix is not None:
                new_args[f"{self.prefix}_{name}"] = value
            else:
                new_args[name] = value

        self.accumulator.update(new_args)
        self.update_medias(new_args)

    def log(
        self,
    ):
        log_dict = self.accumulator.get_metrics_avg()
        if self.sweeping_metric is not None:
            if self.sweeping_metric not in log_dict:
                raise ValueError(
                    f"La metric sur laquelle on sweep: {self.sweeping_metric}, n'existe pas dans les metrics accumulees : {log_dict.keys()}"
                )
            setattr(self, self.sweeping_metric, log_dict[self.sweeping_metric])
            del log_dict[self.sweeping_metric]

            self.wandb_run.log(
                {self.sweeping_metric: getattr(self, self.sweeping_metric)}
            )

        self.wandb_run.log(log_dict)
        self._log_medias()

    def add_media(self, media: WandbMedia):
        self.medias.append(media)

    def update_medias(self, batch: Dict[str, np.ndarray]):
        if self.medias:
            for media in self.medias:
                media.add_batch(batch)

    def _reset_medias(self):
        if self.medias:
            for media in self.medias:
                media.reset()

    def _log_medias(self):
        if self.medias:
            for media in self.medias:
                media.log()

    def finish_run(self):
        self.wandb_run.finish()


if __name__ == "__main__":

    class WandbRunLoggerTester(unittest.TestCase):
        def setUp(self):
            wandb_run = wandb.init(project="test", entity="vincent-coulombe")
            self.logger1 = WandbRunLogger(wandb_run)
            self.logger2 = WandbRunLogger(
                wandb_run,
                medias=[WandbTable("1 colonnes")],
            )

        def test_constructor(self):
            self.assertEqual(len(self.logger1.medias), 0)
            self.assertEqual(len(self.logger2.medias), 1)
            self.assertTrue(isinstance(self.logger2.medias[0], WandbTable))

        def test_add_media(self):
            self.assertEqual(self.logger1.medias, [])
            for i in range(1, 10):
                self.logger1.add_media(WandbTable("test"))
                self.assertTrue(len(self.logger1.medias), i)
                self.assertTrue(isinstance(self.logger1.medias[i - 1], WandbTable))

        def test_update_medias(self):
            self.logger1.add_media(WandbTable("2 colonnes"))
            self.logger1.update_medias(
                {"col1": np.array([1, 2, 3]), "col2": np.array([4, 5, 6])}
            )
            self.assertEqual(
                self.logger1.medias[0].data, {"col1": [1, 2, 3], "col2": [4, 5, 6]}
            )
            self.assertEqual(
                self.logger1.medias[0]._format_data_to_table(),
                (["col1", "col2"], [[1, 4], [2, 5], [3, 6]]),
            )

            self.logger2.update_medias({"col1": np.array([1, 2, 3])})
            self.assertEqual(self.logger2.medias[0].data, {"col1": [1, 2, 3]})
            with self.assertRaises(ValueError):
                self.assertEqual(
                    self.logger2.medias[0]._format_data_to_table(),
                    (["col1"], [[1], [2], [3]]),
                )

        def test_log_medias(self):
            # Test visuel dans le browser
            self.logger1.add_media(WandbTable("3 colonnes"))
            self.logger1.update_medias(
                {
                    "col1": np.array([1, 2, 3]),
                    "col2": np.array([4, 5, 6]),
                    "col3": np.array([7, 8, 9]),
                }
            )
            self.logger1._log_medias()

            self.logger2.update_medias({"col1": np.array([1, 2, 3])})
            self.logger2.update_medias({"col2": np.array([4, 5, 6])})
            self.logger2._log_medias()

        def test_log(self):
            # Test visuel dans le browser
            len_dataset = 5
            for task_id in [1, 2, 3]:
                self.logger1.set_prefix(f"test_task_{task_id}")
                self.logger1.set_sweeping_metric(f"test_task_{task_id}_loss")
                for _ in range(1, 2):
                    self.logger1.reset()
                    for _ in range(1, len_dataset):
                        x = np.random.rand(3) * task_id
                        preds = np.random.rand(3) * task_id
                        y = np.random.rand(3) * task_id
                        loss = randint(1, 10) * task_id
                        score = randint(1, 10) * task_id
                        self.logger1.update(
                            x=x,
                            y=y,
                            preds=preds,
                            loss=loss,
                            score=score,
                        )

                    self.logger1.set_sweeping_metric(f"test_task_{task_id}_loss")
                    self.logger1.log()

            self.logger1.reset()
            self.logger1.set_sweeping_metric(None)
            self.logger1.add_media(WandbTable("Multi-Tasks"))
            len_dataset = 5
            for task_id in [1, 2, 3]:
                self.logger1.set_prefix(f"examine_task_{task_id}")
                self.logger1.accumulator.reset()
                for _ in range(1, len_dataset):
                    x = np.random.rand(3) * task_id
                    preds = np.random.rand(3) * task_id
                    y = np.random.rand(3) * task_id
                    loss = randint(1, 10) * task_id
                    score = randint(1, 10) * task_id
                    self.logger1.update(
                        x=x,
                        y=y,
                        preds=preds,
                        loss=loss,
                        score=score,
                    )
                self.logger1.log()

    unittest.main()
