"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .pipeline import evaluate, featurize, prepare


def load_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the VCG Tucker MI-localization paper")
    parser.add_argument("command", choices=["prepare", "features", "evaluate", "run"])
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--mode", choices=["beat", "patient", "both"], default="both")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command in {"prepare", "run"}:
        prepare(config)
    if args.command in {"features", "run"}:
        print(f"Features saved to {featurize(config)}")
    if args.command in {"evaluate", "run"}:
        modes = None if args.mode == "both" else [args.mode]
        summary = evaluate(config, modes)
        print(json.dumps({key: {"accuracy": value["accuracy"], "macro_roc_auc_ovr": value["macro_roc_auc_ovr"]} for key, value in summary.items()}, indent=2))


if __name__ == "__main__":
    main()

