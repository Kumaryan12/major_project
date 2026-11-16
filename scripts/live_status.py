"""Read-only live process/checkpoint monitor. Ctrl+C stops only this monitor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError): return {}


def process_command(pid):
    """Query native process metadata; never send signals on Windows."""
    try: pid = int(pid)
    except (TypeError, ValueError): return "", "no_pid"
    if pid <= 0: return "", "no_pid"
    try:
        if os.name == "nt":
            code = f"Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' | Select-Object CommandLine | ConvertTo-Json -Compress"
            result = subprocess.run(["powershell", "-NoProfile", "-Command", code], capture_output=True, text=True, timeout=5)
            if result.returncode: return "", "unknown"
            value = json.loads(result.stdout) if result.stdout.strip() else {}
            if not isinstance(value, dict): return "", "checked"
            if value and value.get("CommandLine") is None: return "", "unknown"
            return str(value.get("CommandLine") or ""), "checked"
        result = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=5)
        if result.returncode not in (0, 1): return "", "unknown"
        return result.stdout.strip(), "checked"
    except (OSError, ValueError, subprocess.SubprocessError): return "", "unknown"


def process_status(pid, expected):
    command, state = process_command(pid)
    if state == "unknown": return "unknown"
    if not command: return "not_running"
    return "running" if any(name in command for name in expected) else "pid_reused_or_unrelated"


def age_seconds(value):
    try:
        stamp = datetime.fromisoformat(value["updated_utc"])
        if stamp.tzinfo is None: stamp = stamp.replace(tzinfo=timezone.utc)
        return max(0, int((datetime.now(timezone.utc)-stamp).total_seconds()))
    except (KeyError, ValueError, TypeError): return None


def study_status(root, relative, runner_file, expected, fits, outer):
    target = root/relative;complete = target/"complete"
    progress, queue, summary = read(complete/"progress.json"), read(target/"queue/progress.json"), read(complete/"summary.json")
    runner = read(root/runner_file)
    worker = process_status(progress.get("pid"), expected)
    launcher = process_status(runner.get("runner_pid"), expected)
    done = summary.get("complete") is True and (complete/"REPORT.md").is_file()
    if done: phase = "complete: saved verified report present"
    elif progress.get("state") == "failed" or queue.get("state") == "failed": phase = "FAILED"
    elif worker == "running": phase = progress.get("state", "running")
    elif launcher == "running":
        if queue.get("state") == "waiting_for_signature_verification": phase = "queued: waiting for signature verification"
        elif "--smoke" in queue.get("flags", []): phase = "device smoke check: launcher active"
        elif str(queue.get("module", "")).endswith("_report"): phase = "verification starting: launcher active"
        else: phase = "launcher active: "+str(queue.get("module", queue.get("state", "starting")))
    elif worker == "unknown" or launcher == "unknown": phase = "process status unavailable; inspect saved state"
    elif progress or queue or runner: phase = "not running: saved state only"
    else: phase = "not started on this checkout"
    return {"phase": phase, "completed_fits": len(list(complete.glob("inner_*.json")))+len(list(complete.glob("outer_*.json"))),
            "planned_fits": fits, "completed_outer": len(list(complete.glob("outer_*.json"))) or len(list(complete.glob("fold_*.json"))),
            "planned_outer": outer, "worker_pid": progress.get("pid"), "worker_process": worker,
            "launcher_pid": runner.get("runner_pid"), "launcher_process": launcher,
            "saved_progress_age_seconds": age_seconds(progress), "progress": progress, "queue": queue,
            "report_present": done}


def snapshot(root=ROOT):
    fusion=study_status(root,"results/neural_tensor_fusion_v1","tmp/neural_tensor_fusion_v1_runner.json",
        ["run_neural_tensor_fusion.py","mi_localization.neural_tensor_fusion"],0,10)
    fusion.update(planned_source_contexts=250,source_contexts_verified=fusion["progress"].get("source_contexts_verified",fusion["completed_outer"]*25),new_model_fits=0)
    return {"scope": "Read-only live PID/command checks plus saved checkpoints; does not verify model results or control training.",
            "signature": study_status(root, "results/signature_vcg_v1", "tmp/signature_vcg_v1_runner.json",
                ["run_signature_vcg.sh", "mi_localization.signature_study", "mi_localization.signature_report"], 1050, 150),
            "beat_lengths": study_status(root, "results/neural_chronological_lengths_v1", "tmp/chronological_lengths_v1_runner.json",
                ["run_chronological_lengths.py", "mi_localization.chronology_lengths_study", "mi_localization.chronology_lengths_report"], 1200, 120),
            "residual_context": study_status(root, "results/neural_residual_context_v1", "tmp/residual_context_v1_runner.json",
                ["run_residual_context.py", "mi_localization.residual_context_study", "mi_localization.residual_context_report"], 720, 180),
            "neural_tensor_fusion":fusion}


def display(value):
    india = timezone(timedelta(hours=5, minutes=30), name="IST")
    print("Live experiment status | "+datetime.now(timezone.utc).astimezone(india).strftime("%H:%M:%S IST"))
    for key, title in (("signature", "SIGNATURE STUDY"), ("beat_lengths", "4 / 8 / 16 BEAT STUDY"),
                       ("residual_context", "DIRECT BEATS + GATED CONTEXT"),("neural_tensor_fusion","NEURAL + TENSOR FUSION")):
        row = value[key];p = row["progress"]
        print(f"\n{title}: {row['phase']}")
        if key=="neural_tensor_fusion":
            print(f"  Source contexts {row['source_contexts_verified']}/{row['planned_source_contexts']} | outer folds {row['completed_outer']}/{row['planned_outer']} | new fits: 0")
        else:print(f"  Fits {row['completed_fits']}/{row['planned_fits']} | outer evaluations {row['completed_outer']}/{row['planned_outer']}")
        print(f"  Worker {row['worker_pid']}: {row['worker_process']} | launcher {row['launcher_pid']}: {row['launcher_process']}")
        fields = {k: p[k] for k in ("arm", "training_seed", "sampling_seed", "outer_fold", "inner_fold", "sequence_length", "epoch", "max_epochs", "final_epochs","selected_neural_weight") if k in p}
        if fields: print("  Current: "+", ".join(f"{k}={v}" for k, v in fields.items()))
        if "training_loss" in p: print(f"  Training loss: {p['training_loss']:.4f}")
        if "gate_mean" in p: print(f"  Window loss: {p['window_loss']:.4f} | beat loss: {p['beat_loss']:.4f} | mean gate: {p['gate_mean']:.4f}")
        if "inner_validation_macro_f1" in p: print(f"  Inner-validation F1: {p['inner_validation_macro_f1']:.4f} (not final performance)")
        if row["saved_progress_age_seconds"] is not None: print(f"  Last training/checkpoint update: {row['saved_progress_age_seconds']} s ago")
        if row["phase"] == "FAILED": print("  Error: "+str(p.get("error", row["queue"].get("error", "inspect log"))))
    print("\nCtrl+C closes this monitor only. Experiment processes continue.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true");parser.add_argument("--json", action="store_true")
    parser.add_argument("--interval", type=float, default=5);parser.add_argument("--updates", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not .2 <= args.interval <= 60 or args.updates < 0 or (args.json and args.watch): parser.error("Use interval 0.2–60; JSON is a single snapshot")
    try:
        count = 0
        while True:
            value = snapshot()
            if args.json: print(json.dumps(value, indent=2));return
            if args.watch and sys.stdout.isatty(): print("\033[2J\033[H", end="")
            display(value);count += 1
            if not args.watch or (args.updates and count >= args.updates): break
            time.sleep(args.interval)
    except KeyboardInterrupt: print("\nMonitor closed. Experiments continue.")


if __name__ == "__main__": main()
