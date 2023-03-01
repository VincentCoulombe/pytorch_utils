import torch.nn as nn
from typing import Optional, Dict
import unittest

from model import Model


class FitnessEvaluator(object):
    def __init__(self, objective: Optional[str] = "maximize") -> None:
        if "min" in objective.lower():
            self.maximize = False
        elif "max" in objective.lower():
            self.maximize = True
        else:
            raise ValueError("L'objectif doit être soit de minimizer ou de maximiser")
        self.best_fitness = None

    def save_if_fittest(
        self,
        metrics: Dict[str, float],
        model: Model,
        save_dir: Optional[str] = "checkpoints",
        save_name: Optional[str] = "best.pt",
    ) -> None:
        fitness = next(
            (
                metric_value
                for metric_name, metric_value in metrics.items()
                if "score" in metric_name.lower()
            ),
            None,
        )
        if fitness is None:
            raise ValueError(
                f"Aucune métrique de score n'a été trouvée dans les métriques: {metrics}"
            )
        if self.is_fittest(fitness):
            model.save_checkpoint(save_dir, save_name)

    def is_fittest(self, fitness: float) -> bool:
        if self.best_fitness is None:
            self.best_fitness = fitness
            return True
        elif self.maximize and fitness > self.best_fitness:
            self.best_fitness = fitness
            return True
        elif not self.maximize and fitness < self.best_fitness:
            self.best_fitness = fitness
            return True
        else:
            return False


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
