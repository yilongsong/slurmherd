#!/usr/bin/env python3
"""Report how many epochs a LeRobot run has completed.

    python epochs.py <output_dir> <dataset_repo_id_or_path> [episodes]

Prints a single number on stdout, which is exactly what a slurmherd
``progress.kind: command`` probe consumes. Epochs are not something LeRobot
logs, so they are derived:

    epochs = highest_optimizer_step * batch_size / frames_in_dataset

This runs *on the cluster*, once per reconcile pass. It is dependency-free and
prints ``0`` rather than raising, so a probe never blanks out a whole batch.
"""

from __future__ import annotations

import ast
import json
import os
import sys


def latest_step_and_batch(output_dir: str):
    checkpoints = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(checkpoints):
        return 0, None
    steps = [int(d) for d in os.listdir(checkpoints) if d.isdigit()]
    if not steps:
        return 0, None
    highest = max(steps)
    cfg = os.path.join(
        checkpoints, str(highest), "pretrained_model", "train_config.json"
    )
    try:
        with open(cfg) as fh:
            batch_size = json.load(fh).get("batch_size")
    except (OSError, ValueError):
        batch_size = None
    return highest, batch_size


def total_frames(dataset: str, episodes: str = "") -> int:
    if not os.path.isabs(dataset):
        dataset = os.path.expanduser(f"~/.cache/huggingface/lerobot/{dataset}")
    info_path = os.path.join(dataset, "meta", "info.json")
    try:
        with open(info_path) as fh:
            info = json.load(fh)
    except (OSError, ValueError):
        return 0

    if not episodes:
        return int(info.get("total_frames") or 0)

    try:
        wanted = set(ast.literal_eval(episodes))
    except (ValueError, SyntaxError):
        return int(info.get("total_frames") or 0)

    episodes_path = os.path.join(dataset, "meta", "episodes.jsonl")
    try:
        total = 0
        with open(episodes_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("episode_index") in wanted:
                    total += record.get("length", 0)
        return total or int(info.get("total_frames") or 0)
    except (OSError, ValueError):
        return int(info.get("total_frames") or 0)


def main(argv) -> int:
    if len(argv) < 2:
        print(0)
        return 0
    output_dir = argv[0]
    dataset = argv[1]
    episodes = argv[2] if len(argv) > 2 else ""

    step, batch_size = latest_step_and_batch(output_dir)
    frames = total_frames(dataset, episodes)
    if not step or not batch_size or not frames:
        print(0)
        return 0
    print(round(step * batch_size / frames, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
