"""Smoke, run, verify and optionally publish the selected-convolution comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RESULTS = "results/rocket_tabpfn_v1"


def progress(**values):
    root = ROOT/RESULTS/"queue"
    root.mkdir(parents=True, exist_ok=True)
    path = root/"progress.json.partial"
    path.write_text(json.dumps(dict(pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(), **values), indent=2))
    path.replace(root/"progress.json")


def verified_ready(root):
    target = root/RESULTS/"complete"
    if not all((target/name).is_file() for name in ["summary.json", "REPORT.md", "study_manifest.json"]): return False
    summary = json.loads((target/"summary.json").read_text())
    manifest = json.loads((target/"study_manifest.json").read_text())
    counts = dict(fits=660, all_new_models_restored=660, ridge_models_restored=420, tabpfn_models_restored=240,
        outer_models_restored=120, source_features_reconstructed=120, source_ridge_models_restored=210,
        bootstrap_caches_replayed=120, selector_masks_verified=240)
    return (summary.get("complete") is True and summary.get("study_fingerprint") == manifest.get("study_fingerprint")
            and all(summary.get(k) == v for k, v in counts.items())
            and all(summary.get(k) is True for k in ["source_artifacts_unchanged", "frozen_reference_reproduced", "all_outer_patients_covered"]))


def main():
    from run_residual_context import acquire_lock
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    lock = acquire_lock(ROOT/"tmp/rocket_tabpfn_v1.lock")
    verified = False
    (ROOT/"tmp/rocket_tabpfn_v1_runner.json").write_text(json.dumps(dict(runner_pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat()), indent=2))
    try:
        for module, flags in [("mi_localization.rocket_tabpfn_study", ["--smoke"]),
                              ("mi_localization.rocket_tabpfn_study", []), ("mi_localization.rocket_tabpfn_report", [])]:
            progress(state="running", module=module, flags=flags)
            subprocess.run([sys.executable, "-m", module, *flags], check=True)
        verified = verified_ready(ROOT)
        if not verified: raise RuntimeError("Selected-convolution report verification incomplete")
        progress(state="complete_verified")
        if args.publish:
            branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
            changes = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True)
            if branch != "main" or changes: raise RuntimeError("Verified result ready; branch/tracked files changed before publication")
            subprocess.run(["git", "add", "--", RESULTS], check=True)
            if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode:
                subprocess.run(["git", "commit", "-q", "-m", "Publish verified convolutional feature selection and PFN comparison"], check=True)
            for attempt in range(3):
                if subprocess.run(["git", "push", "origin", "HEAD:main"]).returncode == 0: break
                if attempt == 2: raise RuntimeError("Verified commit exists locally; GitHub push failed")
                time.sleep(10)
            progress(state="complete_published")
    except Exception as error:
        progress(state="publication_failed" if verified else "failed", error=str(error), training_verified=verified)
        raise
    finally: lock.close()


if __name__ == "__main__": main()
