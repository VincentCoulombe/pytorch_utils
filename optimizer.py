import torch
from typing import Union
import unittest


class Optimizer(object):
    SUPPORTED_OPTIMIZERS = ["sgd", "adam", "adamw"]

    def __init__(
        self,
        model: torch.nn.Module,
        lr0: Union[float, int],
        weight_decay: Union[float, int],
        momentum: Union[float, int],
        **kwargs,
    ) -> None:
        self.model = model
        self.lr0 = lr0
        self.weight_decay = weight_decay
        self.momentum = momentum

    def pick_algorithm(self, algorithm: str):
        algorithm = algorithm.lower()
        return getattr(self, algorithm)()

    def sgd(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr0,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=self.momentum > 0.0,
        )

    def adam(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr0,
            weight_decay=self.weight_decay,
            betas=(self.momentum, 0.999),
        )

    def adamw(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr0,
            weight_decay=self.weight_decay,
            betas=(self.momentum, 0.999),
        )


if __name__ == "__main__":

    class OptimizerTester(unittest.TestCase):
        def setUp(self) -> None:
            model = torch.nn.Linear(10, 10)
            self.sgd = Optimizer(model, 0.1, 0.1, 0.1).pick_algorithm("sgd")
            self.adam = Optimizer(model, 0.1, 0.1, 0.1).pick_algorithm("adam")
            self.adamw = Optimizer(model, 0.1, 0.1, 0.1).pick_algorithm("adamw")

        def test_sgd(self):
            self.assertIsInstance(self.sgd, torch.optim.SGD)

        def test_adam(self):
            self.assertIsInstance(self.adam, torch.optim.Adam)

        def test_adamw(self):
            self.assertIsInstance(self.adamw, torch.optim.AdamW)

    unittest.main()
