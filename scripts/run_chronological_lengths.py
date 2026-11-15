"""Queue/run the 4/8/16-beat study and publish only verified aggregates."""
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
RESULTS = "results/neural_chronological_lengths_v1"


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def progress(root, **values):
    target = root/RESULTS/"queue";target.mkdir(parents=True, exist_ok=True)
    temporary = target/"progress.json.partial"
    temporary.write_text(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), **values), indent=2), encoding="utf-8")
    temporary.replace(target/"progress.json")


def dependency_ready(root, fingerprint):
    source = root/"results/signature_vcg_v1/complete"
    if read(source/"study_manifest.json")["study_fingerprint"] != fingerprint:
        raise RuntimeError("Signature dependency fingerprint changed")
    if not (source/"summary.json").is_file() or not (source/"REPORT.md").is_file(): return False
    value = read(source/"summary.json")
    if value.get("study_fingerprint") != fingerprint: raise RuntimeError("Dependency summary differs")
    return (value.get("complete") is True and value.get("all_outer_contexts_restored") is True
            and value.get("all_bootstrap_caches_replayed") is True and value.get("total_contexts") == 1050 and value.get("outer_contexts") == 150)


def wait_for_signature(root):
    source = root/"results/signature_vcg_v1/complete"
    fingerprint = read(source/"study_manifest.json")["study_fingerprint"]
    while not dependency_ready(root, fingerprint):
        status = read(source/"progress.json") if (source/"progress.json").is_file() else {}
        if status.get("state") == "failed": raise RuntimeError("Signature dependency failed")
        runner = root/"tmp/signature_vcg_v1_runner.json"
        if not runner.is_file(): raise RuntimeError("No local signature runner; a clone does not inherit a process")
        if os.name == "posix":
            try: os.kill(int(read(runner)["runner_pid"]), 0)
            except ProcessLookupError:
                if dependency_ready(root, fingerprint): break
                raise RuntimeError("Signature runner ended without a verified report")
        progress(root, state="waiting_for_signature_verification", dependency_fingerprint=fingerprint, dependency_saved_progress=status)
        print("Beat-length study queued: waiting for verified signature results.", flush=True)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument("--wait-for-signature", action="store_true");parser.add_argument("--publish", action="store_true")
    args = parser.parse_args();os.chdir(ROOT)
    try:
        if args.wait_for_signature: wait_for_signature(ROOT)
        for module, flags in (("mi_localization.chronology_lengths_study", ["--smoke"]),
                              ("mi_localization.chronology_lengths_study", []), ("mi_localization.chronology_lengths_report", [])):
            progress(ROOT, state="running", module=module, flags=flags)
            subprocess.run([sys.executable, "-m", module, *flags], check=True)
        progress(ROOT, state="complete_verified")
        if args.publish:
            branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
            changes = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True)
            if branch != "main" or changes: raise RuntimeError("Verified results ready; publication stopped because branch/tracked files changed")
            subprocess.run(["git", "add", "--", RESULTS], check=True)
            if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode:
                subprocess.run(["git", "commit", "-q", "-m", "Publish verified nested beat-length comparison"], check=True)
                subprocess.run(["git", "push", "origin", "HEAD:main"], check=True)
    except Exception as error:
        progress(ROOT, state="failed", error_type=type(error).__name__, error=str(error));raise


if __name__ == "__main__": main()
