import torch
import torch.nn as nn
from torch.cuda import amp
from tqdm import tqdm
import json
import os
import numpy as np
from typing import Dict, Any
import wandb

from datasets.datasets import create_dataloaders
from models.model_picker import pick_model

from logger import WandbRunLogger
from wandb_media import WandbTable
from optimizer import Optimizer
from lr_scheduler import TrainingScheduler
from metrics_accumulator import MetricsAccumulator
from fitness_evaluator import FitnessEvaluator
from task_processor import TaskProcessor


class Teacher(object):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.hyp = config["hyperparameters"]
        self.logging = config["logging"]
        self.system = config["system"]
        self.datasets = config["datasets"]
        self.metrics = config["metrics"]

        # device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.system["cuda"] else "cpu"
        )

        # data
        self.train_dataloader, self.val_dataloader = create_dataloaders(
            **self.datasets, **self.hyp, **self.system
        )

        # model
        self.model = pick_model(
            model_name="efficientnet_v2",
            size="large",
            pretrained=True,
            output_size=2,
        ).to(self.device)

        # optimizer
        self.optimizer = Optimizer(self.model, **self.hyp).pick_algorithm(
            self.hyp["optimizer"]
        )

        # schedule
        warmup_epochs = max(4, self.hyp["epochs"] // 10) if self.hyp["warmup"] else 0
        cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.hyp["epochs"] - warmup_epochs
        )
        self.scheduler = TrainingScheduler(
            self.optimizer,
            self.hyp["lr0"],
            lr_scheduler=cosine_schedule,
            warmup_iteration=warmup_epochs,
        )

        # logs
        self._setup_loggers()

        # fitness evaluator
        self.fitness_evaluator = FitnessEvaluator(objective="min")

        # batch_processor
        self.task_processor = TaskProcessor(
            device=self.device,
            optimizer=self.optimizer,
            task_type=self.metrics["task_type"],
            loss_function=self.metrics["loss_function"],
            scoring_function=self.metrics["scoring_function"],
            mixed_precision=self.hyp["mixed_precision"],
            fitness_evaluator=self.fitness_evaluator,
        )

    def teach(self) -> None:
        for epoch in range(1, self.hyp["epochs"]):
            pbar_desc = f"{epoch}/{self.hyp['epochs']}"

            # Train
            self.model.train()
            if self.train_logger is not None:
                self.train_logger.reset()
            self.task_processor.process_a_task(
                model=self.model,
                dataloader=self.train_dataloader,
                logger=self.train_logger,
                pbar_desc=pbar_desc,
            )

            # Val
            self.model.eval()
            if self.val_logger is not None:
                self.val_logger.reset()
            self.task_processor.process_a_task(
                model=self.model,
                dataloader=self.val_dataloader,
                logger=self.val_logger,
                pbar_desc=pbar_desc,
                ckpt_name="best.pt",
            )

        # Examine best val run
        self.model.from_checkpoint(
            os.path.join(os.path.dirname(__file__), "checkpoints", "best.pt")
        )
        if self.examine_logger is not None:
            self.examine_logger.reset()
        self.task_processor.process_a_task(
            model=self.model,
            dataloader=self.val_dataloader,
            logger=self.examine_logger,
            pbar_desc="Best run:",
        )

    def _setup_loggers(self) -> None:
        wandb_run = wandb.init(
            project=self.logging["project"],
            entity=self.logging["entity"],
            group=self.logging["group"],
            config=self.hyp | self.datasets | self.system,
        )
        if self.logging["enable"]:
            for prefix in ["train", "val", "examine"]:
                setattr(self, WandbRunLogger(wandb_run=wandb_run, prefix=prefix))
            self.val_logger.set_sweeping_metric("val_score")
            self.examine_logger.add_media(WandbTable(title="Best run results"))
        else:
            self.train_logger = None
            self.val_logger = None
            self.examine_logger = None
