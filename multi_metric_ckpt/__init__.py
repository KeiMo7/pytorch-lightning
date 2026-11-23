# Copyright (c) 2026
# Multi-validation-set checkpoint selection, side-mounted on top of Lightning.
"""
Multi-validation-set checkpoint selection
=========================================

A side-mounted callback that tracks the "best" checkpoint independently for
several validation sets (old customers / new customers / long-tail classes /
...), persists every piece of bookkeeping inside the Lightning checkpoint and
keeps a single JSON report as the source of truth shared by the decision,
saving and reporting paths.

The package deliberately lives outside ``lightning.pytorch``: the training
loop, callback registration and the existing ``ModelCheckpoint`` are untouched.
"""

from multi_metric_ckpt.callback import MultiMetricModelCheckpoint, RouteSpec
from multi_metric_ckpt.exceptions import (
    MultiMetricCheckpointError,
    MultiMetricStateMismatchError,
    MultiMetricWriteError,
)

__all__ = [
    "MultiMetricModelCheckpoint",
    "RouteSpec",
    "MultiMetricCheckpointError",
    "MultiMetricStateMismatchError",
    "MultiMetricWriteError",
]
