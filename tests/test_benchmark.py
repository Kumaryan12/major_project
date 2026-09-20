from pathlib import Path

import numpy as np

from mi_localization.benchmark import (
    aggregate_patients,
    create_fixed_folds,
    patient_balanced_weights,
)


def test_patient_balanced_weights_equalize_classes_and_patients():
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    groups = np.array(["a", "a", "b", "c", "d", "d", "d"])
    weights = patient_balanced_weights(y, groups)
    assert np.isclose(weights[y == 0].sum(), weights[y == 1].sum())
    for label in (0, 1):
        totals = [weights[(y == label) & (groups == patient)].sum() for patient in np.unique(groups[y == label])]
        assert np.allclose(totals, totals[0])


def test_patient_aggregation_uses_mean_probability():
    probabilities = np.array([[0.9, 0.1], [0.3, 0.7], [0.2, 0.8]])
    y = np.array([0, 0, 1])
    groups = np.array(["a", "a", "b"])
    ids, truth, result = aggregate_patients(probabilities, y, groups)
    assert ids.tolist() == ["a", "b"]
    assert truth.tolist() == [0, 1]
    assert np.allclose(result[0], [0.6, 0.4])


def test_fixed_folds_keep_patients_whole(tmp_path: Path):
    groups = np.repeat([f"p{i:02d}" for i in range(20)], 3)
    y = np.repeat([0] * 10 + [1] * 10, 3)
    folds = create_fixed_folds(y, groups, ["a", "b"], tmp_path / "folds.csv", 5, 42)
    assert len(folds) == 20
    assert sorted(folds.fold.unique()) == [1, 2, 3, 4, 5]
    assert folds.groupby("fold").label.nunique().eq(2).all()
