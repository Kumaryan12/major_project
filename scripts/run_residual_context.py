"""Run, independently verify and optionally publish the residual-context study."""
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
RESULTS = "results/neural_residual_context_v1"


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def progress(root, **values):
    target = root/RESULTS/"queue";target.mkdir(parents=True, exist_ok=True)
    temporary = target/"progress.json.partial"
    temporary.write_text(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), **values), indent=2), encoding="utf-8")
    temporary.replace(target/"progress.json")


def acquire_lock(path):
    """OS-held lock is released on process exit; no stale PID lock bypass."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0: handle.write(b"0");handle.flush()
            handle.seek(0);msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close();raise RuntimeError("A residual-context launcher already holds the run lock")
    return handle


def verified_ready(root):
    target = root/RESULTS/"complete"
    if not all((target/name).is_file() for name in ("summary.json", "REPORT.md", "study_manifest.json")): return False
    value, scope = read(target/"summary.json"), read(target/"study_manifest.json")
    return (value.get("complete") is True and value.get("study_fingerprint") == scope.get("study_fingerprint")
            and value.get("fits") == 720 and value.get("outer_models_restored") == 180
            and value.get("matched_partition_checks") == 120 and value.get("all_query_beats_covered") is True)


def publish(root):
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    changes = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True)
    if branch != "main" or changes:
        raise RuntimeError("Verified results ready; publication stopped because branch/tracked files changed")
    subprocess.run(["git", "add", "--", RESULTS], cwd=root, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode:
        subprocess.run(["git", "commit", "-q", "-m", "Publish verified morphology-preserving residual-context comparison"], cwd=root, check=True)
    # A commit can exist even if an earlier network push failed; always retry.
    for attempt in range(3):
        result = subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=root)
        if result.returncode == 0: return
        if attempt < 2:
            print("Result push failed; retrying in 10 seconds. Training remains verified.", flush=True);time.sleep(10)
    raise RuntimeError("Verified result commit exists locally, but GitHub push failed three times")


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument("--publish", action="store_true")
    args = parser.parse_args();os.chdir(ROOT)
    lock = acquire_lock(ROOT/"tmp/residual_context_v1.lock")
    (ROOT/"tmp/residual_context_v1_runner.json").write_text(json.dumps({"runner_pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "log": "tmp/residual_context_v1.log", "planned_fits": 720}, indent=2))
    verified = False
    try:
        for module, flags in (("mi_localization.residual_context_study", ["--ensure-smoke"]),
                              ("mi_localization.residual_context_study", []), ("mi_localization.residual_context_report", [])):
            progress(ROOT, state="running", module=module, flags=flags)
            subprocess.run([sys.executable, "-m", module, *flags], check=True)
        verified = verified_ready(ROOT)
        if not verified: raise RuntimeError("Residual-context report did not establish full verification")
        progress(ROOT, state="complete_verified")
        if args.publish:
            publish(ROOT);progress(ROOT, state="complete_published")
    except Exception as error:
        progress(ROOT, state="publication_failed" if verified else "failed", training_verified=verified,
                 error_type=type(error).__name__, error=str(error));raise
    finally:
        lock.close()


if __name__ == "__main__": main()
