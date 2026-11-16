"""Read-only local progress for the four-arm distribution-pooling comparison."""
import argparse
import json
from pathlib import Path
import time
import yaml
from supervised_source_status import command_for_pid


def status(config):
    root = Path(config["output_dir"])/"complete"
    path = root/"progress.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    command = command_for_pid(saved.get("pid"))
    alive = bool(command and any(x in command for x in ("mi_localization.distribution_study", "mi_localization.distribution_report")))
    groups = config["outer_splits"]*len(config["projection_seeds"])
    expected = groups*len(config["arms"])*(config["inner_splits"]*len(config["ridge_alphas"])+1)
    fits = list((root/"review/fits").glob("*.json"))
    contexts = list((root/"review/contexts").glob("*.json"))
    outer = sum(p.name.startswith("outer_") for p in fits)
    lines = [f"Beat-distribution pooling: {root}", f"State: {saved.get('state', 'not started')} | local worker command matches: {alive}",
             f"Updated UTC: {saved.get('updated_utc', 'unavailable')}", f"Saved fit audits: {len(fits)}/{expected}",
             f"Saved outer models: {outer}/{groups*len(config['arms'])}",
             f"Saved feature contexts: {len(contexts)}/{groups*(config['inner_splits']+1)}"]
    if "projection_seed" in saved:
        lines.append(f"Current: projection seed {saved['projection_seed']}, outer fold {saved.get('outer_fold')}, inner {saved.get('inner_fold')}, {saved.get('arm', 'shared features')}")
    lines.extend([f"Final report present: {(root/'REPORT.md').exists()}",
                  "Partial counters/inner scores are not final patient macro F1. All models and transforms require independent replay."])
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--config", default="configs/distribution_pooling_v1.yaml"); p.add_argument("--watch", action="store_true")
    args = p.parse_args(); config = yaml.safe_load(Path(args.config).read_text())
    try:
        while True:
            if args.watch: print("\033[2J\033[H", end="")
            print(status(config), flush=True)
            if not args.watch: break
            time.sleep(10)
    except KeyboardInterrupt: pass
