"""Run nested neural/tensor fusion and publish only its verified aggregates."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RESULTS="results/neural_tensor_fusion_v1"


def progress(**values):
    root=ROOT/RESULTS/"queue";root.mkdir(parents=True,exist_ok=True)
    path=root/"progress.json.partial"
    path.write_text(json.dumps(dict(pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**values),indent=2))
    path.replace(root/"progress.json")


def verified_ready(root):
    p=root/RESULTS/"complete"
    if not (p/"summary.json").is_file() or not (p/"REPORT.md").is_file():return False
    s=json.loads((p/"summary.json").read_text())
    return (s.get("complete") is True and s.get("new_model_fits")==0 and s.get("total_source_contexts_verified")==250
            and s.get("neural_outer_members_restored")==30 and s.get("tensor_outer_contexts_restored")==10
            and s.get("completed_outer_folds")==10 and s.get("frozen_references_reproduced") is True)


def main():
    from run_residual_context import acquire_lock
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--publish",action="store_true")
    args=parser.parse_args();os.chdir(ROOT)
    lock=acquire_lock(ROOT/"tmp/neural_tensor_fusion_v1.lock");verified=False
    (ROOT/"tmp/neural_tensor_fusion_v1_runner.json").write_text(json.dumps(dict(runner_pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(),new_model_fits=0,planned_source_contexts=250),indent=2))
    try:
        for module in ("mi_localization.neural_tensor_fusion","mi_localization.neural_tensor_fusion_report"):
            progress(state="running",module=module,flags=[])
            subprocess.run([sys.executable,"-m",module],check=True)
        verified=verified_ready(ROOT)
        if not verified:raise RuntimeError("Fusion report did not establish all required checks")
        progress(state="complete_verified")
        if args.publish:
            branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip()
            changes=subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True)
            if branch!="main" or changes:raise RuntimeError("Verified fusion ready; publication stopped because branch/tracked files changed")
            subprocess.run(["git","add","--",RESULTS],check=True)
            if subprocess.run(["git","diff","--cached","--quiet"]).returncode:
                subprocess.run(["git","commit","-q","-m","Publish verified patient-nested neural tensor fusion"],check=True)
            for attempt in range(3):
                if subprocess.run(["git","push","origin","HEAD:main"]).returncode==0:break
                if attempt==2:raise RuntimeError("Verified fusion commit exists locally but GitHub push failed")
                time.sleep(10)
            progress(state="complete_published")
    except Exception as error:
        progress(state="publication_failed" if verified else "failed",error=str(error),training_verified=verified);raise
    finally:lock.close()


if __name__=="__main__":main()
