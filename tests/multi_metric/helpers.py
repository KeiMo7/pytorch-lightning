# Copyright (c) 2026
"""Shared fixtures for the multi-metric checkpoint tests."""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningModule, Trainer

from multi_metric_ckpt import MultiMetricModelCheckpoint, RouteSpec


class _ListDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([float(idx)])


class MultiRouteModel(LightningModule):
    """Boring model that logs one configurable score per route per validation.

    ``schedules`` maps route metric key -> list of values indexed by epoch.
    A value of ``None`` means "do not log this metric this epoch" (missing
    route), and ``float("nan")`` / ``inf`` exercise the non-finite path.
    """

    def __init__(
        self,
        schedules: dict[str, list[Optional[float]]],
        train_batches: int = 2,
    ) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self._schedules = schedules
        self._train_batches = train_batches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).mean()
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        epoch = self.current_epoch
        for key, values in self._schedules.items():
            if epoch >= len(values):
                continue
            value = values[epoch]
            if value is None:
                continue
            self.log(key, torch.tensor(float(value), dtype=torch.float32))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def train_dataloader(self):
        return DataLoader(_ListDataset(), batch_size=4)

    def val_dataloader(self):
        return DataLoader(_ListDataset(), batch_size=4)


def make_routes(specs: list[tuple]) -> list[RouteSpec]:
    """``(name, monitor, mode[, patience])`` tuples -> RouteSpec list.

    ``patience`` is passed by keyword because :class:`RouteSpec`'s fourth
    positional field is ``min_delta``; positional passing here would set
    ``min_delta`` to the patience value and silently break comparisons.
    """
    out = []
    for item in specs:
        name, monitor, mode = item[0], item[1], item[2]
        patience = item[3] if len(item) > 3 else 0
        out.append(RouteSpec(name=name, monitor=monitor, mode=mode, patience=patience))
    return out


def make_trainer(tmp_path, callback: MultiMetricModelCheckpoint, max_epochs: int, **kwargs) -> Trainer:
    return Trainer(
        default_root_dir=tmp_path,
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        callbacks=[callback],
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        log_every_n_steps=1,
        **kwargs,
    )


def read_report(callback: MultiMetricModelCheckpoint) -> dict:
    with open(callback.report_path) as fp:
        return json.load(fp)


def list_ckpts(directory: str) -> list[str]:
    return sorted(f for f in os.listdir(directory) if f.endswith(".ckpt"))
