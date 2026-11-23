# Copyright (c) 2026
"""End-to-end demonstration of MultiMetricModelCheckpoint.

Scenarios exercised, mirroring the acceptance criteria:

1. Three validation routes (old/new/long-tail) with independent best and
   patience tracking.
2. An interrupted run resumed from a checkpoint: history and patience carry
   over.
3. The JSON report and the on-disk checkpoints are produced and cross-checked.

Run with:
    PYTHONPATH=src:. python output/demo_e2e.py
"""

import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "multi_metric"))

from lightning.pytorch import Trainer  # noqa: E402

from multi_metric_ckpt import MultiMetricModelCheckpoint, RouteSpec  # noqa: E402
from helpers import MultiRouteModel  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "output", "e2e_run")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

ROUTES = [
    RouteSpec("old_customer", "val/old_auroc", mode="max", patience=2),
    RouteSpec("new_customer", "val/new_auroc", mode="max", patience=2),
    RouteSpec("long_tail", "val/tail_auroc", mode="max", patience=3),
]

# Five epochs of history per route. None = route missing that epoch.
SCHEDULES = {
    "val/old_auroc": [0.50, 0.60, 0.55, 0.54, 0.53],
    "val/new_auroc": [0.40, 0.42, 0.45, None, 0.44],
    "val/tail_auroc": [0.20, 0.21, 0.22, 0.23, 0.24],
}


def make_trainer(callback, max_epochs):
    return Trainer(
        default_root_dir=OUT_DIR,
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        callbacks=[callback],
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        log_every_n_steps=1,
    )


def main():
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ---- phase 1: train two epochs, then "interrupt" ----
    cb1 = MultiMetricModelCheckpoint(routes=ROUTES, dirpath=CKPT_DIR, save_top_k=3, verbose=True)
    make_trainer(cb1, max_epochs=2).fit(MultiRouteModel(SCHEDULES))
    phase1_report = cb1.report()
    resume_ckpt = os.path.join(
        CKPT_DIR, max(phase1_report["checkpoints"], key=lambda r: r["seq"])["filename"]
    )
    print("\n=== Phase 1 (interrupted after epoch 1) ===")
    for name, block in phase1_report["routes"].items():
        print(f"  {name:14s} best={block['best_score']} wait={block['wait_count']}")

    # ---- phase 2: resume into a fresh trainer/callback ----
    cb2 = MultiMetricModelCheckpoint(routes=ROUTES, dirpath=CKPT_DIR, save_top_k=3, verbose=True)
    make_trainer(cb2, max_epochs=5).fit(MultiRouteModel(SCHEDULES), ckpt_path=resume_ckpt)

    report = cb2.report()
    with open(os.path.join(OUT_DIR, "final_report.json"), "w") as fp:
        json.dump(report, fp, indent=2, sort_keys=True)

    print("\n=== Phase 2 (resumed through epoch 4) ===")
    for name, block in report["routes"].items():
        print(
            f"  {name:14s} best={block['best_score']:.3f} @epoch {block['best_epoch']}"
            f" wait={block['wait_count']} available={block['available']}"
            f" best_file={os.path.basename(block['best_model_path'])}"
        )
    print("\nCheckpoints on disk:")
    for f in sorted(os.listdir(CKPT_DIR)):
        print("  ", f)

    on_disk = {f for f in os.listdir(CKPT_DIR) if f.endswith(".ckpt")}
    recorded = {r["filename"] for r in report["checkpoints"] if not r["evicted"]}
    assert on_disk == recorded, "disk and report disagree"
    print("\nOK: report and disk agree; resume preserved history across",
          len(report["routes"]["old_customer"]["history"]), "evaluations.")


if __name__ == "__main__":
    main()
