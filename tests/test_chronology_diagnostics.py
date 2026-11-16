import copy
import json

import numpy as np
import pandas as pd
import pytest

from mi_localization.chronology_diagnostics import (
    Inputs, check_history, correctness_overlap, digest, regroup_probabilities,
    summarize_histories, validate_predictions,
)


def test_record_reweighting_uses_all_beats_then_equal_record_weights():
    # Unequal record lengths can reverse a patient's prediction. Other patients
    # must not affect its result, and beat order must not affect either mean.
    beats = pd.DataFrame({"patient_id": ["a"]*4+["b"], "record": ["a1"]*3+["a2", "b1"]})
    p = np.array([[.8, .2]]*3+[[0., 1.], [.9, .1]])
    weighted = regroup_probabilities(beats, p, ["b", "a"])
    equal = regroup_probabilities(beats, p, ["b", "a"], True)
    np.testing.assert_allclose(weighted, [[.9, .1], [.6, .4]])
    np.testing.assert_allclose(equal, [[.9, .1], [.4, .6]])
    order = [4, 2, 0, 3, 1]
    np.testing.assert_allclose(regroup_probabilities(beats.iloc[order], p[order], ["a", "b"], True), equal[::-1])
    with pytest.raises(ValueError, match="coverage"):
        regroup_probabilities(beats, p, ["a"])
    bad = beats.copy();bad.loc[4, "record"] = "a1"
    with pytest.raises(ValueError, match="boundaries"):
        regroup_probabilities(bad, p, ["a", "b"])
    with pytest.raises(ValueError, match="coverage"):
        regroup_probabilities(beats, p*2, ["a", "b"])


def test_prediction_validation_rejects_leakage_alignment_and_invalid_outputs():
    frozen = pd.DataFrame({"patient_id": ["a", "b"], "label": ["A", "B"], "fold": [1, 2]}).set_index("patient_id")
    frame = pd.DataFrame({"patient_id": ["b", "a"], "truth": ["B", "A"], "fold": [2, 1],
                          "prediction": ["B", "A"], "probability_A": [.1, .9], "probability_B": [.9, .1]})
    np.testing.assert_allclose(validate_predictions(frame, frozen, ["A", "B"], 1.), [[.9, .1], [.1, .9]])
    for column, value in [("fold", 1), ("truth", "A"), ("patient_id", "a"), ("prediction", "A"), ("probability_A", np.nan)]:
        altered = frame.copy();altered.loc[0, column] = value
        with pytest.raises(ValueError):
            validate_predictions(altered, frozen, ["A", "B"], 1.)
    with pytest.raises(ValueError, match="source score"):
        validate_predictions(frame, frozen, ["A", "B"], .5)


def test_error_overlap_keeps_both_wrong_when_wrong_labels_differ():
    truth = np.array([0, 0, 1, 1, 2])
    ref = np.array([0, 0, 0, 2, 2]);seq = np.array([0, 1, 1, 0, 0])
    result = correctness_overlap(truth, ref, seq)
    assert result == {"both_correct": 1, "reference_only_correct": 2, "sequence_only_correct": 1,
                      "both_wrong": 1, "different_predictions": 4}
    assert sum(result[k] for k in ["both_correct", "reference_only_correct", "sequence_only_correct", "both_wrong"]) == len(truth)


def test_history_checks_detect_test_monitoring_and_wrong_stopping():
    fit = {"history": [{"epoch": 1, "training_loss": 1., "validation_patient_macro_f1": .5},
                       {"epoch": 2, "training_loss": .5, "validation_patient_macro_f1": .5}],
           "selected_epoch": 1, "best_validation_macro_f1": .5}
    check_history(fit, 3, 1, True)
    assert summarize_histories([fit], 3)["median_selected_epoch"] == 1
    changed = copy.deepcopy(fit);changed["selected_epoch"] = 2
    with pytest.raises(ValueError, match="stopping"):
        check_history(changed, 3, 1, True)
    changed = copy.deepcopy(fit);changed["history"][1]["epoch"] = 3
    with pytest.raises(ValueError, match="history"):
        check_history(changed, 3, 1, True)
    with pytest.raises(ValueError, match="Outer"):
        check_history(fit, 3, 1, False)
    outer = {"history": [{"epoch": 1, "training_loss": .7}], "selected_epoch": 1, "best_validation_macro_f1": None}
    check_history(outer, 3, 1, False)


def test_manifest_tampering_and_reproducible_provenance(tmp_path):
    path = tmp_path/"manifest.json";body = {"config": {"seed": 42}}
    path.write_text(json.dumps(dict(body, study_fingerprint=digest(body))))
    inputs = Inputs();assert inputs.manifest(path)["config"]["seed"] == 42
    original = inputs.hashes[str(path)]
    altered = dict(body, config={"seed": 123}, study_fingerprint=digest(body))
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="manifest"):
        inputs.manifest(path)
    assert inputs.hashes[str(path)] != original
