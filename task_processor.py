import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda import amp
from typing import Optional
from tqdm import tqdm
import os

from loss_function import LossFunction
from scoring_function import ScoringFunction
from predictions_getter import PredictionsGetter
from fitness_evaluator import FitnessEvaluator
from logger import Logger


class TaskProcessor(object):
    def __init__(
        self,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        task_type: str,
        loss_function: str,
        scoring_function: str,
        mixed_precision: Optional[bool] = True,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
    ) -> None:
        self.device = device
        self.fp16 = bool(self.device != "cpu" and mixed_precision)
        self.scaler = amp.GradScaler(enabled=self.fp16)
        self.optimizer = optimizer
        self.fitness_evaluator = fitness_evaluator
        self.predictions_getter = PredictionsGetter(task_type)
        self.scoring_function = ScoringFunction(scoring_function)
        self.loss_function = LossFunction(loss_function)

    def process_a_task(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        logger: Optional[Logger] = None,
        pbar_description: Optional[str] = None,
        ckpt_name: Optional[str] = "best.pt",
    ):
        phase = "Train" if model.training else "Val"
        fp16 = self.scaler is not None
        pbar = tqdm(dataloader, total=len(dataloader))

        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            with amp.autocast(enabled=fp16):
                with torch.set_grad_enabled(phase == "Train"):
                    logits = self(x)
                    loss = self.loss_function(logits, y)

            x, y, logits = (
                self._tensor_to_numpy(x),
                self._tensor_to_numpy(y),
                self._tensor_to_numpy(logits),
            )
            loss = loss.item()
            predictions = self.predictions_getter(logits)
            score = self.scoring_function(predictions, y)

            if logger is not None:
                logger.update(
                    x=x,
                    preds=predictions,
                    y=y,
                    loss=loss,
                    score=score,
                )

            if phase == "Train":
                self.optimizer.zero_grad()
                if fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            self._update_pbar_description(pbar, pbar_description)

        if logger is not None:
            logger.log()
            if phase == "Val" and isinstance(self.fitness_evaluator, FitnessEvaluator):
                self.fitness_evaluator.save_if_fittest(
                    logger.accumulator.get_metrics_avg(),
                    model,
                    save_dir=os.path.join(os.path.dirname(__file__), "checkpoints"),
                    save_name=ckpt_name,
                )

    @staticmethod
    def _update_pbar_description(pbar: tqdm, pbar_description: str):
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.2f}"
        s = f"{pbar_description}, {mem}GB"
        pbar.set_description(s)

    @staticmethod
    def _tensor_to_cpu(tensor: torch.Tensor):
        return tensor.detach().cpu()

    def _tensor_to_numpy(self, tensor: torch.Tensor):
        return self._tensor_to_cpu(tensor).numpy()

    def _tensor_to_list(self, tensor: torch.Tensor):
        return self._tensor_to_numpy(tensor).tolist()
