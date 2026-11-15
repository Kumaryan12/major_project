"""Reference-disjoint, signed multiscale deviations from measured XYZ medoids."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pywt
from scipy.fft import dct

from .benchmark import inner_patient_splits
from .multiview_bank import array_hash
from .neural import _hash


def check_partition(ids, train, reference, query):
    """No classifier row, reference row or held-out query may share a patient."""
    parts = [np.asarray(rows, dtype=int) for rows in (train, reference, query)]
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Expected exactly one row per patient")
    for rows in parts:
        if rows.ndim != 1 or len(rows) != len(np.unique(rows)) or np.any(rows < 0) or np.any(rows >= len(ids)):
            raise ValueError("Invalid patient partition")
    if any(set(ids[a]) & set(ids[b]) for a, b in combinations(parts, 2)):
        raise ValueError("Reference/classifier/query patient overlap")


def reference_members(y, ids, training, query, seed, parts=3):
    training, query = np.asarray(training), np.asarray(query)
    check_partition(ids, training, [], query)
    members = []
    for fit, ref in inner_patient_splits(y[training], ids[training], parts, seed):
        fit, ref = training[fit], training[ref]
        check_partition(ids, fit, ref, query)
        if set(fit) | set(ref) != set(training):
            raise ValueError("Incomplete reference partition")
        members.append((fit, ref))
    counts = np.zeros(len(ids), dtype=int)
    for fit, _ in members:
        counts[fit] += 1
    if np.any(counts[training] != parts-1):
        raise ValueError("Unequal classifier participation")
    return members


def exact_two_medoids(X):
    """Globally minimize sum of nearest squared Euclidean distances; stable ties."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or len(X) < 2 or not np.isfinite(X).all():
        raise ValueError("Need two finite reference patients")
    distance = np.maximum(np.sum((X[:, None]-X[None, :])**2, axis=-1), 0)
    pairs = list(combinations(range(len(X)), 2))
    costs = [float(np.minimum(distance[:, i], distance[:, j]).sum()) for i, j in pairs]
    return np.asarray(pairs[int(np.argmin(costs))], dtype=int)


def fit_atlas(patient_waves, y, ids, train, reference, query, hc, arm, seed, coefficients=32):
    check_partition(ids, train, reference, query)
    if arm not in {"healthy", "mixed"}:
        raise ValueError("Unknown reference arm")
    reference = np.asarray(reference)
    eligible = reference[y[reference] == hc]
    if len(eligible) < 2:
        raise ValueError("Too few healthy reference patients")
    if arm == "mixed":
        # Reference labels determine the shared size only, not mixed identities.
        eligible = np.sort(np.random.default_rng(seed).choice(reference, len(eligible), replace=False))
    eligible = eligible[np.argsort(ids[eligible], kind="stable")]
    coefficients_array = dct(patient_waves[eligible], type=2, axis=-1, norm="ortho")[..., :coefficients].reshape(len(eligible), -1)
    chosen = eligible[exact_two_medoids(coefficients_array)]
    waves = np.asarray(patient_waves[chosen], dtype=np.float64)
    return waves, {
        "arm": arm, "reference_patients": len(reference), "eligible_reference_patients": len(eligible),
        "reference_membership_sha256": _hash(ids[reference].tolist()),
        "eligible_membership_sha256": _hash(ids[eligible].tolist()),
        "eligible_waveforms_sha256": array_hash(patient_waves[eligible]),
        "medoid_membership_sha256": _hash(ids[chosen].tolist()), "medoid_waveforms_sha256": array_hash(waves),
        "metric": "squared Euclidean, signed XYZ DCT32 in physical mV, exact two-medoids",
        "self_reference_overlap": 0,
    }


def feature_names(source):
    bands = [f"A{source['level']}"] + [f"D{i}" for i in range(source["level"], 0, -1)]
    return [f"reference_{k}/{axis}/{band}/{window}/{stat}" for k in range(2)
            for axis in ("X", "Y", "Z") for band in bands for window in source["windows"]
            for stat in ("mean", "rms")] + ["reference_0/distance", "reference_1/distance", "nearest_distance_margin"]


def deviation_features(record_waves, record_patients, query_ids, atlas, source, coefficients=32):
    """Pure query transform: labels never enter. Equal-record coordinate median."""
    query_ids = np.asarray(query_ids)
    if not len(query_ids) or len(np.unique(query_ids)) != len(query_ids):
        raise ValueError("Require distinct query patients")
    if not set(query_ids).issubset(set(record_patients)):
        raise ValueError("Missing query patient's recording templates")
    rows = np.flatnonzero(np.isin(record_patients, query_ids))
    x = np.asarray(record_waves[rows], dtype=np.float64)
    if atlas.shape != (2, 3, x.shape[-1]) or not np.isfinite(x).all() or not np.isfinite(atlas).all():
        raise ValueError("Invalid measured XYZ query/reference waves")
    delta = x[:, None] - atlas[None]
    tensor = np.stack(pywt.mra(delta, source["wavelet"], level=source["level"],
                              transform="dwt", mode="symmetric", axis=-1), axis=-2)
    if not np.allclose(tensor.sum(-2), delta, rtol=1e-8, atol=1e-8):
        raise ValueError("Deviation tensor failed additive reconstruction")
    cells = []
    for start, stop in source["windows"].values():
        window = tensor[..., start:stop]
        cells.append(np.stack([window.mean(-1), np.sqrt(np.mean(window**2, axis=-1))], axis=-1))
    summaries = np.stack(cells, axis=-2).reshape(len(x), -1)
    coeff = dct(delta, type=2, axis=-1, norm="ortho")[..., :coefficients]
    distances = np.sqrt(np.sum(coeff**2, axis=(-2, -1)))
    margin = np.abs(distances[:, 0]-distances[:, 1])[:, None]
    record_features = np.concatenate([summaries, distances, margin], axis=1)
    result = []
    for patient in query_ids:
        local = record_features[record_patients[rows] == patient]
        if not len(local):
            raise ValueError("Missing query patient's recording templates")
        result.append(np.median(local, axis=0))
    result = np.asarray(result)
    if result.shape != (len(query_ids), len(feature_names(source))) or not np.isfinite(result).all():
        raise ValueError("Invalid deviation feature dimensions/content")
    return result
