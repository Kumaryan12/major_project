"""Dependency-free, read-only orientation for a fresh or existing checkout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {
    "raw_ptb_manifest": "data/raw/ptbdb/manifest.csv",
    "beat_table": "data/processed/beats.csv",
    "relational_statistics": "data/processed/relational_vcg_v1/statistics_short.npy",
    "denoised_waveforms": "data/processed/neural_vcg_v1/waveforms.npy",
    "raw_waveforms": "data/processed/neural_raw_v1/waveforms.npy",
    "recording_templates": "data/processed/multiview_vcg_v1/templates.npy",
    "patient_tensor_features": "data/processed/tensor_features_v1/patient_features.npz",
    "refined_tensor_features": "data/processed/tensor_refinement_v1/patient_features.npz",
    "ordered_signature_features": "data/processed/signature_vcg_v1/patient_features.npz",
    "chronological_rr_metadata": "data/processed/chronological_vcg_v1/metadata.npz",
    "foundation_checkpoint": "data/models/tabpfn_v2/tabpfn-v2-classifier-v2_default.ckpt",
}
REPORTS = {
    "presentation_notes": "docs/implementation_presentation_notes.md",
    "resnet_seed_repeatability": "results/neural_repeat_v1/REPORT.md",
    "bootstrap_feature_consensus": "results/feature_stability_v1/complete/REPORT.md",
    "tabpfn_v2": "results/tabpfn_v2_v1/complete/REPORT.md",
    "healthy_reference_deviation": "results/deviation_vcg_v1/complete/REPORT.md",
    "ordered_signatures": "results/signature_vcg_v1/complete/REPORT.md",
    "consecutive_beats_rr": "results/neural_chronological_v1/complete/REPORT.md",
    "nested_beat_lengths": "results/neural_chronological_lengths_v1/complete/REPORT.md",
}


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"read_error": "ExpectedJSONObject"}
    except (OSError, ValueError) as error:
        return {"read_error": type(error).__name__}


def inventory(root=ROOT):
    root = Path(root)
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                capture_output=True, text=True, timeout=10, check=False)
        revision = result.stdout.strip() if result.returncode == 0 else "unavailable"
    except (OSError, subprocess.SubprocessError):
        revision = "unavailable"
    study = root/"results/deviation_vcg_v1/complete"
    summary_path, progress_path = study/"summary.json", study/"progress.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    progress = read_json(progress_path) if progress_path.exists() else None
    signature = root/"results/signature_vcg_v1/complete"
    signature_summary = read_json(signature/"summary.json") if (signature/"summary.json").is_file() else {}
    chronology = root/"results/neural_chronological_v1/complete"
    chronology_summary = read_json(chronology/"summary.json") if (chronology/"summary.json").is_file() else {}
    queue_progress = root/"results/neural_chronological_v1/queue/progress.json"
    lengths = root/"results/neural_chronological_lengths_v1/complete"
    length_summary = read_json(lengths/"summary.json") if (lengths/"summary.json").is_file() else {}
    length_queue = root/"results/neural_chronological_lengths_v1/queue/progress.json"
    return {
        "repository_revision": revision,
        "interpreter": sys.version.split()[0],
        "scope": "Read-only file inventory. No training, downloads, numerical imports or independent result verification.",
        "read_first": ["AGENTS.md", "docs/CODEX_HANDOFF.md", "docs/WINDOWS_SETUP.md"],
        "local_artifacts": {name: {"path": path, "present": (root/path).is_file()} for name, path in ARTIFACTS.items()},
        "report_files": {name: {"path": path, "present": (root/path).is_file()} for name, path in REPORTS.items()},
        "healthy_reference_study": {
            "study_manifest_present": (study/"study_manifest.json").is_file(),
            "completed_member_audit_files": len(list(study.glob("inner_*.json")))+len(list(study.glob("outer_*.json"))),
            "planned_member_contexts": 1960,
            "outer_evaluation_files": len(list(study.glob("fold_*.json"))),
            "planned_outer_evaluations": 100,
            "completed_summary_and_report_present": summary.get("complete") is True and (study/"REPORT.md").is_file(),
            "saved_progress": progress,
            "live_process_status": "Not checked. Saved files/PIDs cannot prove a process is alive, especially on another computer.",
            "fresh_clone_note": "Missing local study artifacts are expected. Aryan's running process and ignored checkpoints do not transfer through Git.",
        },
        "ordered_signature_study": {
            "study_manifest_present": (signature/"study_manifest.json").is_file(),
            "completed_context_audit_files": len(list(signature.glob("inner_*.json")))+len(list(signature.glob("outer_*.json"))),
            "planned_contexts": 1050,
            "outer_evaluation_files": len(list(signature.glob("fold_*.json"))),
            "planned_outer_evaluations": 150,
            "completed_summary_and_report_present": signature_summary.get("complete") is True and (signature/"REPORT.md").is_file(),
            "saved_progress": read_json(signature/"progress.json") if (signature/"progress.json").is_file() else None,
            "live_process_status": "Not checked. Saved progress does not establish a live process or transfer it to another computer.",
        },
        "chronological_study": {
            "scope": "Superseded fixed-length proposal; no full fit was started",
            "completed_fit_audit_files": len(list(chronology.glob("inner_*.json")))+len(list(chronology.glob("outer_*.json"))),
            "planned_fits": 480, "outer_evaluation_files": len(list(chronology.glob("outer_*.json"))), "planned_outer_evaluations": 120,
            "completed_summary_and_report_present": chronology_summary.get("complete") is True and (chronology/"REPORT.md").is_file(),
            "saved_training_progress": read_json(chronology/"progress.json") if (chronology/"progress.json").is_file() else None,
            "saved_queue_progress": read_json(queue_progress) if queue_progress.is_file() else None,
            "live_process_status": "Not checked. A saved queue, active training and a completed result are different states.",
        },
        "chronological_length_study": {
            "candidate_lengths": [4, 8, 16], "planned_fits": 1200, "planned_outer_evaluations": 120,
            "completed_fit_audit_files": len(list(lengths.glob("inner_*.json")))+len(list(lengths.glob("outer_*.json"))),
            "outer_evaluation_files": len(list(lengths.glob("outer_*.json"))),
            "completed_summary_and_report_present": length_summary.get("complete") is True and (lengths/"REPORT.md").is_file(),
            "saved_training_progress": read_json(lengths/"progress.json") if (lengths/"progress.json").is_file() else None,
            "saved_queue_progress": read_json(length_queue) if length_queue.is_file() else None,
            "live_process_status": "Not checked by this inventory. Use python scripts/live_status.py --watch for live PID/command checks.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the same inventory as JSON")
    args = parser.parse_args()
    value = inventory()
    if args.json:
        print(json.dumps(value, indent=2, ensure_ascii=True))
        return
    print(f"Project checkout: {value['repository_revision']} | Python {value['interpreter']}")
    print(value["scope"])
    print("Read first: "+", ".join(value["read_first"]))
    print("\nLocal artifacts (not included in Git):")
    for item in value["local_artifacts"].values():
        print(f"  {'present' if item['present'] else 'missing'}: {item['path']}")
    print("\nAggregate report files (presence is not verification):")
    for item in value["report_files"].values():
        print(f"  {'present' if item['present'] else 'missing'}: {item['path']}")
    study = value["healthy_reference_study"]
    print(f"\nHealthy-reference saved member audits: {study['completed_member_audit_files']}/{study['planned_member_contexts']}")
    print(f"Saved outer evaluations: {study['outer_evaluation_files']}/{study['planned_outer_evaluations']}")
    print(f"Completed summary and report present: {study['completed_summary_and_report_present']}")
    if study["saved_progress"] is not None:
        print("Last saved progress: "+json.dumps(study["saved_progress"], ensure_ascii=True))
    print(study["live_process_status"])
    print(study["fresh_clone_note"])
    signature = value["ordered_signature_study"]
    print(f"\nOrdered-signature saved contexts: {signature['completed_context_audit_files']}/{signature['planned_contexts']}")
    print(f"Saved outer evaluations: {signature['outer_evaluation_files']}/{signature['planned_outer_evaluations']}")
    print(f"Completed summary and report present: {signature['completed_summary_and_report_present']}")
    if signature["saved_progress"] is not None:
        print("Last saved progress: "+json.dumps(signature["saved_progress"], ensure_ascii=True))
    print(signature["live_process_status"])
    chronology = value["chronological_study"]
    print(f"\nSuperseded fixed-length saved fits: {chronology['completed_fit_audit_files']}/{chronology['planned_fits']}")
    print(f"Completed summary and report present: {chronology['completed_summary_and_report_present']}")
    for key in ("saved_queue_progress", "saved_training_progress"):
        if chronology[key] is not None: print(key+": "+json.dumps(chronology[key], ensure_ascii=True))
    print(chronology["live_process_status"])
    lengths = value["chronological_length_study"]
    print(f"\nCurrent 4/8/16-length saved fits: {lengths['completed_fit_audit_files']}/{lengths['planned_fits']}")
    print(f"Completed summary and report present: {lengths['completed_summary_and_report_present']}")
    for key in ("saved_queue_progress", "saved_training_progress"):
        if lengths[key] is not None: print(key+": "+json.dumps(lengths[key], ensure_ascii=True))
    print(lengths["live_process_status"])


if __name__ == "__main__":
    main()
