"""Native annotation semantics, source-role isolation and read-only audit integrity."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mi_localization import supervised_source_audit as audit


def tiny_config(tmp_path):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/supervised_source_audit_v1.yaml").read_text())
    for key, name in [("metadata_path", "metadata.csv"), ("statements_path", "statements.csv"),
                      ("target_beats_path", "beats.csv"), ("target_folds_path", "folds.csv")]:
        config[key] = str(tmp_path / name)
    config.update(waveform_dir=str(tmp_path / "source"), target_waveform_dir=str(tmp_path / "target"),
                  output_dir=str(tmp_path / "audit"), minimum_training_positive_patients=2,
                  validation_support_caution_below=2, expected_target=dict(patients=2, records=2, beats=4))
    for key in ["prior_cohort_path", "prior_cohort_manifest", "prior_screen_manifest", "prior_screen_summary"]:
        config[key] = str(tmp_path / key)
    rows = []
    for fold in range(1, 11):
        for offset in [1, 2]:
            ecg_id = (fold - 1)*2+offset
            rows.append(dict(ecg_id=ecg_id, patient_id=ecg_id, strat_fold=fold,
                filename_hr=f"records500/00000/{ecg_id:05d}_hr",
                scp_codes="{'AMI':100,'ALMI':100,'NORM':0}" if offset == 1 else "{'NORM':100,'AMI':0,'SR':0}"))
    rows.append(dict(ecg_id=21, patient_id=1, strat_fold=1, filename_hr="records500/00000/00021_hr", scp_codes="{'AMI':50}"))
    database = pd.DataFrame(rows)
    # Excluded labels must never reach parsing, even if malformed.
    database.loc[database.strat_fold.eq(10), "scp_codes"] = "NOT A DICTIONARY"
    database.to_csv(config["metadata_path"], index=False)
    codes = config["review_codes"]+["LMI", "SR"]
    statements = pd.DataFrame(index=codes, data=dict(description=codes,
        diagnostic=[1]*7+[0], diagnostic_class=["MI"]*5+["NORM", "MI", None],
        diagnostic_subclass=["AMI", "AMI", "AMI", "IMI", "IMI", "NORM", "LMI", None]))
    statements.to_csv(config["statements_path"])
    pd.DataFrame(dict(patient_id=["target_a", "target_a", "target_b", "target_b"],
                      record=["a", "a", "b", "b"], label=["AMI", "AMI", "HC", "HC"])).to_csv(config["target_beats_path"], index=False)
    pd.DataFrame(dict(patient_id=["target_a", "target_b"], label=["AMI", "HC"], fold=[0, 1])).to_csv(config["target_folds_path"], index=False)
    Path(config["target_waveform_dir"]).mkdir()
    (Path(config["target_waveform_dir"]) / "a.dat").write_bytes(bytes(120000))
    (Path(config["target_waveform_dir"]) / "b.dat").write_bytes(b"different target encoding")
    for key in ["metadata", "statements", "target_beats", "target_folds"]:
        config[f"{key}_sha256"] = audit.file_hash(config[f"{key}_path"])
    return config, database, statements


def make_pair(config, ecg_id, units="mV", data=True, content=None, fmt="16"):
    path = Path(config["waveform_dir"]) / f"records500/00000/{ecg_id:05d}_hr"
    path.parent.mkdir(parents=True, exist_ok=True)
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    header = [f"{path.name} 12 500 5000"]
    header += [f"{path.name}.dat {fmt} 1000/{units} 16 0 0 0 0 {lead}" for lead in leads]
    path.with_suffix(".hea").write_text("\n".join(header)+"\n")
    if data:
        path.with_suffix(".dat").write_bytes(bytes(120000) if content is None else content)
    return path


def test_native_states_preserve_overlap_unknown_intermediate_and_visits(tmp_path):
    config, database, statements = tiny_config(tmp_path)
    frame = audit.source_frame(database, config)
    labels, codes, states, likelihood = audit.label_states(frame, statements, config)
    assert len(frame) == 19 and frame.patient_id.nunique() == 18
    assert "SR" not in labels and "HC" not in labels
    a, b = labels.index("AMI"), labels.index("ALMI")
    one, two, visit = [frame.index[frame.ecg_id.eq(i)][0] for i in [1, 2, 21]]
    assert states[one, a] == states[one, b] == 3
    assert states[two, a] == 1 and likelihood[two, a] == 0
    assert states[visit, a] == 2 and likelihood[visit, a] == 50
    assert states[visit, b] == 0 and np.isnan(likelihood[visit, b])
    summary = audit.combination_summary(frame, codes, states, labels, statements)
    assert summary["train"]["multiple_100_mi_codes"]["patients"] == 8
    assert summary["train"]["patients_with_changing_listed_diagnostic_sets"] == 1
    assert not frame.strat_fold.eq(10).any()


def test_validation_changes_cannot_choose_training_support_flags(tmp_path):
    config, database, statements = tiny_config(tmp_path)
    frame = audit.source_frame(database, config)
    labels, _, states, _ = audit.label_states(frame, statements, config)
    first = audit.label_summary(frame, labels, states, config)
    changed = states.copy()
    changed[frame.role.eq("validation")] = 3
    second = audit.label_summary(frame, labels, changed, config)
    assert {k:v["training_support_eligible"] for k,v in first.items()} == {k:v["training_support_eligible"] for k,v in second.items()}
    assert not first["LMI"]["training_support_eligible"]
    assert first["AMI"]["validation_support_caution"] and not second["AMI"]["validation_support_caution"]


def test_annotation_parser_rejects_ambiguous_or_invalid_likelihoods():
    for value in ["{'AMI':0,'AMI':100}", "{'AMI':True}", "{'AMI':-1}", "{'AMI':101}",
                  "{'AMI':1e309}", "{'other':100}", "[]", "{1:100}", "{'AMI':None}", None]:
        with pytest.raises(ValueError):
            audit.parse_codes(value, {"AMI"})
    assert audit.parse_codes("{'AMI':0}", {"AMI"}) == {"AMI": 0.0}


def test_identity_path_and_role_validation_precede_waveform_access(tmp_path, monkeypatch):
    config, database, _ = tiny_config(tmp_path)
    bad = database.copy();bad.loc[2, "patient_id"] = 1
    with pytest.raises(ValueError, match="crosses folds"):
        audit.source_frame(bad, config)
    bad = database.copy();bad.loc[0, "filename_hr"] = "../excluded"
    with pytest.raises(ValueError, match="Unsafe"):
        audit.source_frame(bad, config)
    bad = database.copy();bad.loc[0, "ecg_id"] = 2
    with pytest.raises(ValueError, match="Duplicate"):
        audit.source_frame(bad, config)
    frame = audit.source_frame(database, config).iloc[:1].copy()
    frame["strat_fold"] = 10
    monkeypatch.setattr(audit, "safe_path", lambda *args: pytest.fail("accessed excluded waveform"))
    with pytest.raises(ValueError, match="Refusing"):
        audit.source_inventory(frame, config)


def test_header_inventory_and_encoded_duplicates_do_not_clear_identity(tmp_path, monkeypatch):
    config, database, statements = tiny_config(tmp_path)
    for ecg_id in [1, 3, 17, 19]:
        make_pair(config, ecg_id)
    make_pair(config, 2, units="uV", content=b"x"*120000)
    make_pair(config, 18, data=False)
    make_pair(config, 5, fmt="24", content=b"y"*120000)
    make_pair(config, 7, content=b"short")
    original_hash = audit.file_hash
    def guarded_hash(path):
        assert "00019_hr" not in str(path) and "00020_hr" not in str(path)
        return original_hash(path)
    monkeypatch.setattr(audit, "file_hash", guarded_hash)
    monkeypatch.setattr(audit.wfdb, "rdrecord", lambda *a, **k: pytest.fail("decoded waveform"))
    inventory, availability = audit.source_inventory(audit.source_frame(database, config), config)
    assert availability["train"]["cached_complete_pairs"] == 5
    assert availability["train"]["compatible_header_and_size_records"] == 2
    assert availability["validation"]["cached_complete_pairs"] == 1
    assert availability["validation"]["compatible_header_and_size_records"] == 1
    assert inventory.set_index("ecg_id").loc[2, "header_status"] == "non_mv_units"
    assert inventory.set_index("ecg_id").loc[5, "header_status"] == "unsupported_encoding_or_data_path"
    frame = audit.source_frame(database, config)
    labels, _, states, _ = audit.label_states(frame, statements, config)
    cached = audit.cached_label_summary(frame, labels, states, inventory.sample(frac=1, random_state=3))
    assert cached["ALMI"]["train"]["patients"] == 2 and cached["ALMI"]["validation"]["patients"] == 1
    assert cached["NORM"]["train"]["patients"] == 0
    target, _ = audit.target_inventory(config)
    counts, groups, matches = audit.duplicate_summary(inventory, target)
    assert counts["cross_role_duplicate_groups"] == counts["cross_patient_duplicate_groups"] == 1
    assert counts["exact_encoded_source_target_match_records"] == 3
    assert set(matches) == {1, 3, 17} and len(groups) == 1
    assert not counts["patient_identity_independence_cleared"]
    assert not counts["waveform_near_duplicate_screen_performed"]


def test_safe_paths_refuse_symlinks_even_within_dataset_root(tmp_path):
    root = tmp_path / "root";root.mkdir()
    (root / "excluded.dat").write_bytes(b"excluded")
    (root / "eligible.dat").symlink_to(root / "excluded.dat")
    for path in ["../outside", str(tmp_path / "absolute"), "eligible.dat"]:
        with pytest.raises(ValueError):
            audit.safe_path(root, path)
    assert audit.safe_path(root, "missing.dat") == root / "missing.dat"


def test_full_audit_roundtrip_privacy_and_frozen_scope(tmp_path, monkeypatch):
    config, _, _ = tiny_config(tmp_path)
    make_pair(config, 1)
    monkeypatch.setattr(audit.wfdb, "rdrecord", lambda *a, **k: pytest.fail("decoded waveform"))
    result = audit.run(config)
    assert audit.run(config, verify=True) == result == audit.run(config)
    assert result["complete"] and result["source_only_metadata_feasible"]
    for key in ["clean_target_transfer_ready", "training_launched", "classification_result_available",
                "patient_identity_independence_cleared", "native_code_target_equivalence_clinically_reviewed",
                "unlisted_is_adjudicated_negative", "expanded_source_waveform_qc_complete"]:
        assert result[key] is False
    assert result["excluded_fold_10"]["labels_parsed"] == result["excluded_fold_10"]["waveform_files_accessed"] == 0
    root = Path(config["output_dir"])
    with np.load(root / "review/native_label_states.npz") as bank:
        assert len(bank["ecg_id"]) == 19 and 19 not in bank["ecg_id"] and 20 not in bank["ecg_id"]
        assert bank["states"].shape == (19, 7)
    public = "\n".join(p.read_text() for p in root.iterdir() if p.is_file())
    assert "target_a" not in public and "00001_hr" not in public
    compatibility = pd.read_csv(root / "compatibility_review.csv")
    assert not compatibility.clinical_equivalence_approved.any()
    source_dat = Path(config["waveform_dir"]) / "records500/00000/00001_hr.dat"
    source_dat.write_bytes(b"changed")
    with pytest.raises(ValueError, match="inputs/code/cache changed"):
        audit.run(config, verify=True)


def test_output_metadata_and_protected_target_tampering_fail(tmp_path):
    config, _, _ = tiny_config(tmp_path)
    audit.run(config)
    root = Path(config["output_dir"])
    report = root / "REPORT.md";saved = report.read_bytes()
    report.write_text("changed")
    with pytest.raises(ValueError, match="output changed"):
        audit.run(config, verify=True)
    report.write_bytes(saved)
    with pytest.raises(ValueError, match="official metadata checksum"):
        audit.analyze(dict(config, metadata_sha256="0"*64))
    Path(config["target_folds_path"]).write_text("changed")
    with pytest.raises(ValueError, match="Protected target"):
        audit.analyze(config)


def test_historical_screen_linkage_is_not_expanded_waveform_clearance(tmp_path):
    config, database, _ = tiny_config(tmp_path)
    pd.DataFrame(dict(ecg_id=[1, 17])).to_csv(config["prior_cohort_path"], index=False)
    Path(config["prior_cohort_manifest"]).write_text(json.dumps(dict(cohort_sha256=audit.file_hash(config["prior_cohort_path"]))))
    Path(config["prior_screen_manifest"]).write_text(json.dumps(dict(screen_fingerprint="old-scope")))
    Path(config["prior_screen_summary"]).write_text(json.dumps(dict(screen_fingerprint="old-scope",
        source_records=2, continuous_pairs_checked=8, positive_controls=2, recording_screen_passed=True)))
    result = audit.prior_evidence(audit.source_frame(database, config), config)
    assert result["candidate_records_in_historical_cohort"] == 2
    assert result["historical_screen_passed"] and result["historical_only"]
    assert not result["expanded_cohort_screened"] and not result["patient_identity_independence_cleared"]
    Path(config["prior_screen_manifest"]).write_text(json.dumps(dict(screen_fingerprint="wrong-scope")))
    with pytest.raises(ValueError, match="scope mismatch"):
        audit.prior_evidence(audit.source_frame(database, config), config)
    Path(config["prior_cohort_path"]).write_text("changed")
    with pytest.raises(ValueError, match="cohort was modified"):
        audit.prior_evidence(audit.source_frame(database, config), config)
