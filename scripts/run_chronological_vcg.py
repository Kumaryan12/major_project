"""Run chronology directly or queue it behind the current local signature study."""
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


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def progress(root, **values):
    target = root/"results/neural_chronological_v1/queue"
    target.mkdir(parents=True, exist_ok=True)
    temporary = target/"progress.json.partial"
    temporary.write_text(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), **values), indent=2), encoding="utf-8")
    temporary.replace(target/"progress.json")


def dependency_ready(root, fingerprint):
    study = root/"results/signature_vcg_v1/complete"
    if read_json(study/"study_manifest.json")["study_fingerprint"] != fingerprint:
        raise RuntimeError("Signature dependency manifest changed while queued")
    if not (study/"REPORT.md").is_file() or not (study/"summary.json").is_file():
        return False
    summary = read_json(study/"summary.json")
    if summary.get("study_fingerprint") != fingerprint:
        raise RuntimeError("Signature dependency report fingerprint differs")
    return (summary.get("complete") is True and summary.get("all_outer_contexts_restored") is True
            and summary.get("all_bootstrap_caches_replayed") is True and summary.get("total_contexts") == 1050 and summary.get("outer_contexts") == 150)


def wait_for_signature(root):
    study = root/"results/signature_vcg_v1/complete"
    fingerprint = read_json(study/"study_manifest.json")["study_fingerprint"]
    while not dependency_ready(root, fingerprint):
        status = read_json(study/"progress.json") if (study/"progress.json").is_file() else {}
        if status.get("state") == "failed":
            raise RuntimeError("Signature dependency failed; queued chronology was not started")
        runner = root/"tmp/signature_vcg_v1_runner.json"
        if not runner.is_file():
            raise RuntimeError("No local signature-runner record; do not infer a remote process from Git")
        if os.name == "posix":
            try:
                os.kill(int(read_json(runner)["runner_pid"]), 0)
            except ProcessLookupError:
                if dependency_ready(root, fingerprint): break
                raise RuntimeError("Local signature runner ended without a verified dependency report")
        progress(root, state="waiting_for_signature_verification", dependency_fingerprint=fingerprint,
                 dependency_saved_progress=status, live_status="Local POSIX runner checked where supported; saved progress is not a remote process monitor")
        print("Chronology queued: waiting for complete verified signature report.", flush=True)
        time.sleep(30)
    return fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-signature", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    try:
        if args.wait_for_signature: wait_for_signature(ROOT)
        for module, flags in (("mi_localization.chronology_study", ["--smoke"]),
                              ("mi_localization.chronology_study", []), ("mi_localization.chronology_report", [])):
            progress(ROOT, state="running", module=module, flags=flags)
            subprocess.run([sys.executable, "-m", module, *flags], check=True)
        progress(ROOT, state="complete_verified")
        if args.publish:
            branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
            changes = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True)
            if branch != "main" or changes:
                raise RuntimeError("Verified chronology results are ready; publication stopped because branch/tracked files changed")
            subprocess.run(["git", "add", "--", "results/neural_chronological_v1"], check=True)
            if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode:
                subprocess.run(["git", "commit", "-q", "-m", "Publish verified chronological VCG and RR comparisons"], check=True)
                subprocess.run(["git", "push", "origin", "HEAD:main"], check=True)
    except Exception as error:
        progress(ROOT, state="failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
