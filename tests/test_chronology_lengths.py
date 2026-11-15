import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from mi_localization import chronology_lengths_study as study, chronology_lengths_report as report
from mi_localization import chronology_model as models
from test_chronology import fixture


def script(name):
    path = Path(__file__).resolve().parents[1]/"scripts"/name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_default_grid_matches_exposure_and_steps_without_changing_model_shape():
    root = Path(__file__).resolve().parents[1]
    grid = yaml.safe_load((root/"configs/chronological_lengths_v1.yaml").read_text())
    base = yaml.safe_load((root/grid["base_config"]).read_text())
    assert grid["lengths"] == [4, 8, 16]
    configs = [study.context_config(grid, base, k) for k in grid["lengths"]]
    assert [c["windows_per_epoch"] for c in configs] == [2048, 1024, 512]
    assert [c["batch_size"] for c in configs] == [32, 16, 8]
    assert [c["inference_batch_size"] for c in configs] == [64, 32, 16]
    assert {c["windows_per_epoch"]//c["batch_size"] for c in configs} == {64}
    shapes = [[tuple(p.shape) for p in models.initial_model(c, 42, "ordered_rr").parameters()] for c in configs]
    assert shapes[0] == shapes[1] == shapes[2]
    with pytest.raises(ValueError, match="divide"):
        study.context_config(dict(grid, beat_slots_per_epoch=8193), base, 4)


def test_symmetric_inner_choice_and_configured_ties():
    values = {"ordered_rr": {"4": {"mean_inner_macro_f1": .8}, "8": {"mean_inner_macro_f1": .7}, "16": {"mean_inner_macro_f1": .7}},
              "shuffled_rr": {"4": {"mean_inner_macro_f1": .1}, "8": {"mean_inner_macro_f1": .7}, "16": {"mean_inner_macro_f1": .7}}}
    chosen, rows = study.choose_length([4, 8, 16], values)
    assert chosen == 8 and rows[0]["paired_inner_mean"] == pytest.approx(.45)
    swapped = {"ordered_rr": values["shuffled_rr"], "shuffled_rr": values["ordered_rr"]}
    assert study.choose_length([4, 8, 16], swapped)[0] == chosen


def test_nested_length_study_actual_training_restoration_resume_and_tamper(tmp_path, monkeypatch):
    bank, assigned, base = fixture(tmp_path)
    grid = dict(base_config="test", output_dir=str(tmp_path/"lengths"), lengths=[2, 4],
                beat_slots_per_epoch=8, beat_slots_per_batch=4, inference_beat_slots=8, length_selection="paired_inner_mean")
    monkeypatch.setattr(study, "load_inputs", lambda config: (base, bank, assigned))
    root = study.run_study(grid)
    summary = report.summarize_study(grid)
    assert summary["fits"] == 40 and summary["outer_models_restored"] == 8
    assert summary["matched_gru_partition_checks"] == 10
    assert sum(summary["shared_selected_length_counts"].values()) == 2
    for fold in (1, 2):
        decision = json.loads((root/f"selection_{fold:02d}_s42.json").read_text())
        paths = list(root.glob(f"outer_{fold:02d}_*.json"))
        assert len(paths) == 4
        assert all(json.loads(p.read_text())["sequence_length"] == decision["selected_length"] for p in paths)
    monkeypatch.setattr(models, "fit", lambda *a, **k: pytest.fail("Completed length fits must resume"))
    assert study.run_study(grid) == root and report.summarize_study(grid) == summary
    with pytest.raises(ValueError, match="changed"):
        study.run_study(dict(grid, beat_slots_per_epoch=16))
    path = root/"selection_01_s42.json"
    value = json.loads(path.read_text());value["selected_length"] = 999;path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="choice changed"):
        report.summarize_study(grid)


def test_live_monitor_distinguishes_pid_reuse_queue_training_and_completion(tmp_path, monkeypatch):
    live = script("live_status.py")
    target = tmp_path/"results/neural_chronological_lengths_v1"
    complete = target/"complete";complete.mkdir(parents=True)
    queue = target/"queue";queue.mkdir()
    tmp = tmp_path/"tmp";tmp.mkdir()
    commands = {}
    monkeypatch.setattr(live, "process_command", lambda pid: (commands.get(pid, ""), "checked"))
    assert "not started" in live.snapshot(tmp_path)["beat_lengths"]["phase"]
    (tmp/"chronological_lengths_v1_runner.json").write_text(json.dumps({"runner_pid": 11}))
    commands[11] = "python scripts/run_chronological_lengths.py --wait-for-signature"
    (queue/"progress.json").write_text(json.dumps({"state": "waiting_for_signature_verification", "pid": 11}))
    assert "queued" in live.snapshot(tmp_path)["beat_lengths"]["phase"]
    (queue/"progress.json").write_text(json.dumps({"state": "running", "module": "mi_localization.chronology_lengths_study", "flags": ["--smoke"]}))
    assert "smoke" in live.snapshot(tmp_path)["beat_lengths"]["phase"]
    (queue/"progress.json").write_text(json.dumps({"state": "running", "module": "mi_localization.chronology_lengths_report", "flags": []}))
    assert "verification starting" in live.snapshot(tmp_path)["beat_lengths"]["phase"]
    (complete/"progress.json").write_text(json.dumps({"pid": 12, "state": "inner_training", "sequence_length": 8, "epoch": 2}))
    commands[12] = "python -m mi_localization.chronology_lengths_study"
    assert live.snapshot(tmp_path)["beat_lengths"]["phase"] == "inner_training"
    commands[12] = "python unrelated.py";commands[11] = "python another.py"
    result = live.snapshot(tmp_path)["beat_lengths"]
    assert result["worker_process"] == "pid_reused_or_unrelated" and "not running" in result["phase"]
    (complete/"summary.json").write_text(json.dumps({"complete": True}));(complete/"REPORT.md").write_text("verified")
    assert live.snapshot(tmp_path)["beat_lengths"]["report_present"]


def test_new_queue_requires_full_verified_dependency(tmp_path):
    queue = script("run_chronological_lengths.py")
    root = tmp_path/"results/signature_vcg_v1/complete";root.mkdir(parents=True)
    (root/"study_manifest.json").write_text(json.dumps({"study_fingerprint": "source"}))
    assert not queue.dependency_ready(tmp_path, "source")
    (root/"REPORT.md").write_text("done")
    (root/"summary.json").write_text(json.dumps({"complete": True, "study_fingerprint": "source", "total_contexts": 1050,
        "outer_contexts": 150, "all_outer_contexts_restored": True, "all_bootstrap_caches_replayed": True}))
    assert queue.dependency_ready(tmp_path, "source")
    with pytest.raises(RuntimeError, match="fingerprint"):
        queue.dependency_ready(tmp_path, "changed")
