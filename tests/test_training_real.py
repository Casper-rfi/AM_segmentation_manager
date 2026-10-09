"""One real training run through AM-model-training, which holds the two packages together.

Off unless ``AM_SEG_TEST_TRAINING=1``: one epoch over one image is minutes on a CPU,
the model being generation 1's at its full 1536 pixels. It refines random weights, so
nothing is downloaded, and passes only if am-seg can load what the run wrote.
"""
from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from am_segmentation_manager.ml import environment

#: Set to 1 to run this module.
ENV_VAR = "AM_SEG_TEST_TRAINING"

pytestmark = [
    pytest.mark.skipif(os.environ.get(ENV_VAR) != "1", reason=f"{ENV_VAR}=1 runs a real training"),
    pytest.mark.skipif(not environment.training_stack_present(), reason="train extra not installed"),
]

#: Long enough for a two-core runner's CPU.
TIMEOUT_SECONDS = 45 * 60


def test_a_run_trains_and_am_seg_loads_what_it_wrote(tmp_path: Path):
    import numpy as np
    import tifffile
    import torch

    from am_segmentation_manager.ml.model import GENERATION_1, build_architecture, load_model
    from am_segmentation_manager.models import ImageRecord
    from am_segmentation_manager.operations.csv import export_training_csv
    from am_segmentation_manager.operations.training_process import run_status, start_training
    from am_segmentation_manager.operations.training_run import (
        REFINE,
        apply_training_run,
        plan_training_run,
    )
    from am_segmentation_manager.training_runs import find_training_runs, read_training_run

    collection = tmp_path / "collection"
    rng = np.random.default_rng(0)
    records = []
    for index in range(2):
        stage = collection / f"{index:02d}-lamella" / "Adaptive_polish_1"
        image = stage / "sem" / f"AP_img_{index:03d}_SEM.tif"
        label = stage / "labels" / image.name
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        tifffile.imwrite(image, rng.integers(0, 65535, (192, 256), dtype=np.uint16))
        classes = np.zeros((192, 256), np.uint8)
        classes[:30], classes[40:150, 30:220], classes[150:160, 100:140], classes[160:] = 1, 2, 3, 4
        tifffile.imwrite(label, classes)
        records.append(
            ImageRecord(image_path=image, labels=[label], state="corrected", modality="SEM")
        )
    export_training_csv(records, collection, tmp_path / "export", prefix="set", train_ratio=0.5)
    base = tmp_path / "base.pth"
    torch.save(build_architecture(GENERATION_1).state_dict(), base)

    plan = plan_training_run(
        tmp_path / "export" / "set_manifest.yaml", tmp_path / "runs", "real",
        replace(REFINE, max_epochs=1, training_batch_size=1, save_all=True),
        base_weights=base, device="cpu",
    )
    run = apply_training_run(plan)
    start_training(run)

    deadline = time.monotonic() + TIMEOUT_SECONDS
    while run_status(run).state == "running" and time.monotonic() < deadline:
        time.sleep(5)
    status = run_status(run)
    assert status.state == "finished", (status, (run / "train.log").read_text(encoding="utf-8")[-3000:])

    (written,) = find_training_runs(run, depth=0)
    trained = read_training_run(written)
    assert trained.finished and len(trained.train) == 1 and len(trained.val) == 1
    (checkpoint,) = trained.checkpoints
    assert checkpoint.whole()
    assert checkpoint.path.name == "real_val_epoch001.pth"

    loaded = load_model(checkpoint.path)
    assert loaded.description.generation.name == "1.0"
    assert loaded.predict(np.asarray(tifffile.imread(records[0].image_path))).shape == (1152, 1536)


def test_a_checkpoint_scored_per_image_matches_am_model_training_s_own_evaluation(tmp_path: Path):
    """Random weights over three small images, once through each, with nothing trained."""
    import json

    import numpy as np
    import tifffile
    import torch
    from am_model_training.main import evaluate

    from am_segmentation_manager.io.tables import write_csv
    from am_segmentation_manager.ml.model import GENERATION_1, build_architecture
    from am_segmentation_manager.ml.scoring import score_checkpoint
    from am_segmentation_manager.validation_scores import ImageScores, metrics_for

    rng = np.random.default_rng(0)
    rows = []
    for index in range(3):
        image, label = tmp_path / f"image{index}.tif", tmp_path / f"label{index}.tif"
        tifffile.imwrite(image, rng.integers(0, 65535, (192, 256), dtype=np.uint16))
        classes = np.zeros((192, 256), np.uint8)
        classes[:30], classes[40:150, 30:220] = 1, 2
        if index:
            classes[150:160, 100:140], classes[160:] = 3, 4
        tifffile.imwrite(label, classes)
        rows.append((str(image), str(label)))
    listed = tmp_path / "held.csv"
    write_csv(listed, ("image", "label"), rows)
    weights = tmp_path / "random_epoch001.pth"
    torch.manual_seed(0)
    torch.save(build_architecture(GENERATION_1).state_dict(), weights)

    evaluate(tmp_path / "theirs", listed, weights, cpu_only=True)
    (reported_file,) = (tmp_path / "theirs").rglob("*_eval_metrics.json")
    reported = json.loads(reported_file.read_text(encoding="utf-8"))

    matrices, losses = score_checkpoint(weights, listed, device="cpu")
    ours = metrics_for(ImageScores(
        1, tuple(image for image, _ in rows),
        tuple(tuple(tuple(row) for row in matrix) for matrix in matrices), tuple(losses),
    ))

    assert ours.loss == pytest.approx(reported["loss"], rel=1e-4)
    for name in ("iou", "dice", "f1", "precision", "recall"):
        assert ours.per_class[name] == pytest.approx(reported[name], abs=1e-5, nan_ok=True), name
        average = f"weighted_average_{name}"
        assert ours.scalars[average] == pytest.approx(reported[average], abs=1e-5), average
