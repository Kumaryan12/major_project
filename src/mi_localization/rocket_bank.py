"""Per-record coordinate-median templates from every original denoised beat."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from . import chronology_bank
from .contrastive import _progress
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash


@dataclass
class TemplateBank:
    templates: np.ndarray
    record_patients: np.ndarray
    ids: np.ndarray
    labels: np.ndarray
    provenance: dict


def prepare(config):
    original = chronology_bank.prepare_bank(config)
    root = Path(config["template_bank_dir"])
    root.mkdir(parents=True, exist_ok=True)
    audit = Path(config["output_dir"])/"bank_audit"
    audit.mkdir(parents=True, exist_ok=True)
    scope = {"source": original.provenance, "code_sha256": file_hash(__file__), "numpy": np.__version__,
             "rule": "Every retained denoised beat enters its own recording's coordinate median; no beat selection, alignment, or unit normalization"}
    path, guard = root/"templates.npz", root/"manifest.json"
    if guard.exists():
        manifest = json.loads(guard.read_text())
        if manifest["scope"] != scope or manifest["templates_sha256"] != file_hash(path):
            raise ValueError("Recording template source/content changed; use a new bank namespace")
        with np.load(path) as saved:
            templates = saved["templates"]
    else:
        values = []
        for i in range(len(original.record_patients)):
            values.append(np.median(np.asarray(original.waves[original.rows(i)]), axis=0))
            _progress(audit, state="building_templates", records_complete=i+1,
                      total_records=len(original.record_patients))
        templates = np.stack(values).astype(np.float32)
        temporary = path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, templates=templates)
        temporary.replace(path)
        manifest = {"scope": scope, "templates_sha256": file_hash(path), "n_beats": len(original.waves),
                    "n_records": len(templates), "n_patients": len(original.ids), "shape": list(templates.shape),
                    "record_membership_sha256": array_hash(original.record_patients)}
        _save_json(guard, manifest)
    if (templates.shape != (len(original.record_patients), 3, original.waves.shape[-1])
            or not np.isfinite(templates).all() or manifest["record_membership_sha256"] != array_hash(original.record_patients)):
        raise ValueError("Template membership/shape/values changed")
    provenance = {"bank_manifest": manifest, "manifest_sha256": file_hash(guard)}
    _save_json(audit/"summary.json", manifest)
    _progress(audit, state="complete", records_complete=len(templates), total_records=len(templates))
    return TemplateBank(templates, original.record_patients, original.ids, original.labels, provenance)
