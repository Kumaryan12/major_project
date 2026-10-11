"""VCG-only relational tensors with training-fold-only shared projections.

All region boundaries are R-relative hypotheses, not validated delineations.
The previously inspected PTB folds remain a development benchmark.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import yaml
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score

from .benchmark import (
    aggregate_patients, inner_patient_splits, load_or_create_folds,
    make_estimator, metric_bundle, patient_balanced_weights,
    predict_probabilities, save_confusion,
)
from .data import read_vcg
from .features import _canonicalize_columns, wavelet_statistical_features
from .signal import denoise_vcg


ARMS = (
    "statistics_short", "statistics_context", "correlation_flat",
    "correlation_shared", "covariance_shared",
)


def recording_tensor(signal: np.ndarray, wavelet: str, level: int,
                     detail_levels: tuple[int, ...]) -> np.ndarray:
    """Compute lead x recording-time x band values before cropping beats."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != 3 or not np.isfinite(signal).all():
        raise ValueError("Expected finite recording-time x 3 measured VCG")
    if level > pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len):
        raise ValueError("Recording is too short for the requested context wavelet level")
    if not detail_levels or min(detail_levels) < 1 or max(detail_levels) > level:
        raise ValueError("Detail levels must be within the decomposition level")
    tensor = np.empty((3, len(signal), len(detail_levels) + 1), dtype=np.float32)
    tensor[:, :, 0] = signal.T
    for lead in range(3):
        coefficients = pywt.wavedec(signal[:, lead], wavelet, level=level, mode="symmetric")
        for band, detail in enumerate(detail_levels, 1):
            selected = [np.zeros_like(c) for c in coefficients]
            selected[1 + level - detail] = coefficients[1 + level - detail]
            tensor[lead, :, band] = pywt.waverec(selected, wavelet, mode="symmetric")[:len(signal)]
    return tensor


def pack_symmetric(matrices: np.ndarray) -> np.ndarray:
    """Pack (..., regions, channels, channels), region-major upper triangles."""
    matrices = np.asarray(matrices)
    if matrices.ndim < 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("Expected square matrices with an explicit region axis")
    indices = np.triu_indices(matrices.shape[-1])
    values = matrices[..., indices[0], indices[1]]
    return values.reshape(*matrices.shape[:-3], -1)


def unpack_symmetric(packed: np.ndarray, channels: int, regions: int) -> np.ndarray:
    packed = np.asarray(packed)
    triangle = channels * (channels + 1) // 2
    if packed.shape[-1] != triangle * regions:
        raise ValueError("Packed matrix dimensions do not match channels/regions")
    values = packed.reshape(*packed.shape[:-1], regions, triangle)
    matrices = np.zeros((*packed.shape[:-1], regions, channels, channels), dtype=packed.dtype)
    row, col = np.triu_indices(channels)
    matrices[..., row, col] = values
    matrices[..., col, row] = values
    return matrices


def relational_matrices(tensor: np.ndarray, windows: list[list[int]],
                        before_r: int, shrinkage: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return region x channel x channel correlation and covariance matrices.

    Channels are XYZ, each with original + reconstructed wavelet details.
    Flat channels have zero correlation (including their self-correlation)
    before fixed diagonal shrinkage. No statistics are fitted across patients.
    """
    tensor = np.asarray(tensor, dtype=np.float64)
    if tensor.ndim != 3 or not np.isfinite(tensor).all():
        raise ValueError("Expected a finite lead x time x band tensor")
    if not 0 <= shrinkage <= 1:
        raise ValueError("Shrinkage must be in [0, 1]")
    channels = tensor.transpose(0, 2, 1).reshape(-1, tensor.shape[1])
    correlations, covariances = [], []
    for start, stop in windows:
        left, right = before_r + int(start), before_r + int(stop)
        if not 0 <= left < right <= tensor.shape[1] or right - left < 2:
            raise ValueError("R-relative region is outside the beat or shorter than 2 samples")
        segment = channels[:, left:right]
        centered = segment - segment.mean(axis=1, keepdims=True)
        covariance = centered @ centered.T / (segment.shape[1] - 1)
        norms = np.sqrt(np.maximum(np.diag(covariance), 0))
        denominator = np.outer(norms, norms)
        correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance),
                                where=denominator > 1e-20)
        correlation = np.clip(correlation, -1, 1)
        correlations.append((1 - shrinkage) * correlation + shrinkage * np.eye(len(channels)))
        covariances.append((1 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance)))
    if not correlations:
        raise ValueError("At least one region is required")
    return np.stack(correlations), np.stack(covariances)


class SharedRelationalProjection(TransformerMixin, BaseEstimator):
    """Learn one symmetric channel basis, retaining the physical region axis.

    The weighted second-moment objective maximizes reconstruction energy, not
    diagnostic accuracy. Rank is selected inside patient-grouped validation.
    Covariance channel scaling is also learned exclusively in fit().
    """

    def __init__(self, rank: int = 8, channels: int = 24, regions: int = 4,
                 statistics: int = 192, covariance: bool = False, chunk_size: int = 1024):
        self.rank = rank
        self.channels = channels
        self.regions = regions
        self.statistics = statistics
        self.covariance = covariance
        self.chunk_size = chunk_size

    def _validate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        width = self.channels * (self.channels + 1) // 2 * self.regions + self.statistics
        if X.ndim != 2 or X.shape[1] != width or not np.isfinite(X).all():
            raise ValueError(f"Expected finite samples x {width} features")
        if self.chunk_size < 1 or not 1 <= self.rank <= self.channels:
            raise ValueError("Invalid projection rank or chunk size")
        return X

    def _matrices(self, X: np.ndarray) -> np.ndarray:
        width = self.channels * (self.channels + 1) // 2 * self.regions
        matrices = unpack_symmetric(X[:, :width].astype(np.float64), self.channels, self.regions)
        return matrices / self.channel_scale_[None, None, :, None] / self.channel_scale_[None, None, None, :]

    def fit(self, X: np.ndarray, y=None, sample_weight: np.ndarray | None = None):
        X = self._validate(X)
        if len(X) == 0:
            raise ValueError("Cannot fit a projection without training samples")
        weights = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights.shape != (len(X),) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Invalid sample weights")
        weights = weights / weights.sum()
        self.n_features_in_ = X.shape[1]
        self.n_fit_samples_ = len(X)
        self.channel_scale_ = np.ones(self.channels)
        if self.covariance:
            row, col = np.triu_indices(self.channels)
            diagonal = np.flatnonzero(row == col)
            width = len(row) * self.regions
            variance = np.zeros(self.channels)
            for start in range(0, len(X), self.chunk_size):
                stop = min(start + self.chunk_size, len(X))
                packed = X[start:stop, :width].reshape(-1, self.regions, len(row))
                variance += np.einsum("n,nri->i", weights[start:stop], packed[:, :, diagonal]) / self.regions
            self.channel_scale_ = np.sqrt(np.maximum(variance, 1e-12))
        moment = np.zeros((self.channels, self.channels))
        for start in range(0, len(X), self.chunk_size):
            stop = min(start + self.chunk_size, len(X))
            matrices = self._matrices(X[start:stop])
            weighted = matrices * np.sqrt(weights[start:stop, None, None, None] / self.regions)
            unfolded = weighted.transpose(2, 0, 1, 3).reshape(self.channels, -1)
            moment += unfolded @ unfolded.T
        eigenvalues, eigenvectors = np.linalg.eigh(moment)
        self.basis_ = _canonicalize_columns(eigenvectors[:, ::-1][:, :self.rank])
        self.energy_fraction_ = float(np.maximum(eigenvalues[-self.rank:], 0).sum() / max(eigenvalues.clip(0).sum(), 1e-30))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, ["basis_", "channel_scale_"])
        X = self._validate(X)
        width = self.rank * (self.rank + 1) // 2 * self.regions
        result = np.empty((len(X), width + self.statistics), dtype=np.float32)
        for start in range(0, len(X), self.chunk_size):
            stop = min(start + self.chunk_size, len(X))
            matrices = self._matrices(X[start:stop])
            core = np.einsum("ia,nrij,jb->nrab", self.basis_, matrices, self.basis_, optimize=True)
            result[start:stop, :width] = pack_symmetric(core)
        if self.statistics:
            result[:, width:] = X[:, -self.statistics:]
        return result


def _record_rows(record: str, rows: pd.DataFrame, paper: dict, config: dict):
    raw, fs = read_vcg(Path(paper["data"]["raw_dir"]), record, list(paper["data"]["leads"]))
    if fs != int(paper["data"]["sampling_rate"]):
        raise ValueError(f"Unexpected sampling rate for {record}")
    clean = denoise_vcg(raw, paper["signal"]["denoise_wavelet"], int(paper["signal"]["denoise_level"]))
    wave = paper["features"]
    full = recording_tensor(clean, wave["wavelet"], int(wave["level"]), tuple(wave["detail_levels"]))
    before, after = int(paper["data"]["before_r"]), int(paper["data"]["after_r"])
    outputs = {name: [] for name in ["statistics_context", "correlation", "covariance"]}
    for peak in rows.r_peak.astype(int):
        if peak - before < 0 or peak + after >= len(clean):
            raise ValueError(f"Incomplete cached beat in {record}")
        beat = full[:, peak - before:peak + after + 1, :]
        correlation, covariance = relational_matrices(beat, config["windows"], before, float(config["shrinkage"]))
        outputs["statistics_context"].append(wavelet_statistical_features(beat))
        outputs["correlation"].append(pack_symmetric(correlation))
        outputs["covariance"].append(pack_symmetric(covariance))
    return rows.index.to_numpy(), {key: np.vstack(value).astype(np.float32) for key, value in outputs.items()}


def _fingerprint(config: dict, paper: dict, beats: pd.DataFrame) -> str:
    sources = []
    for record in beats.record.unique():
        for extension in ["hea", "xyz"]:
            path = Path(paper["data"]["raw_dir"]) / f"{record}.{extension}"
            stat = path.stat()
            sources.append((str(path), stat.st_size, stat.st_mtime_ns))
    original = Path(config["original_bank_path"])
    stat = original.stat()
    payload = {"schema": 1, "paper": paper, "windows": config["windows"],
               "classes": config["classes"], "shrinkage": config["shrinkage"],
               "beats_sha256": hashlib.sha256(Path(config["beats_path"]).read_bytes()).hexdigest(),
               "original_bank": (str(original), stat.st_size, stat.st_mtime_ns), "raw_sources": sources}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_relational_bank(config: dict) -> Path:
    with Path(config["paper_config"]).open() as stream:
        paper = yaml.safe_load(stream)
    all_beats = pd.read_csv(config["beats_path"])
    mask = all_beats.label.isin(config["classes"])
    selected_indices = np.flatnonzero(mask)
    beats = all_beats.loc[mask].reset_index(drop=True)
    if beats.empty:
        raise ValueError("The configured relational cohort is empty")
    fingerprint = _fingerprint(config, paper, beats)
    destination = Path(config["bank_dir"])
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Feature sources/settings changed; use a new bank_dir instead of overwriting the cache")
        for key in ["statistics_short", "statistics_context", "correlation", "covariance", "y", "groups"]:
            array = np.load(destination / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            if array.shape[0] != len(beats):
                raise ValueError("Cached bank is incomplete")
        return destination
    with np.load(config["original_bank_path"], allow_pickle=False) as original:
        if not np.array_equal(original["groups"], all_beats.patient_id.astype(str).to_numpy()) or not np.array_equal(original["y"], all_beats.label.astype(str).to_numpy()):
            raise ValueError("Original feature bank does not align with the beat manifest")
        short = original["wavelet_statistics"][selected_indices]
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "statistics_short.npy", short)
    np.save(destination / "y.npy", beats.label.astype(str).to_numpy(dtype="U"))
    np.save(destination / "groups.npy", beats.patient_id.astype(str).to_numpy(dtype="U"))
    channels = 3 * (len(paper["features"]["detail_levels"]) + 1)
    regions = len(config["windows"])
    widths = {"statistics_context": channels * 8,
              "correlation": regions * channels * (channels + 1) // 2,
              "covariance": regions * channels * (channels + 1) // 2}
    outputs = {key: np.lib.format.open_memmap(destination / f"{key}.npy", mode="w+", dtype=np.float32,
                                           shape=(len(beats), width)) for key, width in widths.items()}
    tasks = Parallel(n_jobs=int(config.get("n_jobs", 2)), return_as="generator_unordered")(
        delayed(_record_rows)(record, rows, paper, config) for record, rows in beats.groupby("record", sort=False))
    for completed, (indices, values) in enumerate(tasks, 1):
        for key, matrix in values.items():
            outputs[key][indices] = matrix
        if completed % 25 == 0 or completed == beats.record.nunique():
            print(f"Relational features: {completed}/{beats.record.nunique()} records", flush=True)
    for matrix in outputs.values():
        matrix.flush()
        if not np.isfinite(matrix).all():
            raise ValueError("Generated feature bank contains nonfinite values")
    manifest = {"schema": 1, "fingerprint": fingerprint, "n_beats": len(beats),
                "n_patients": beats.patient_id.nunique(), "n_records": beats.record.nunique(),
                "channels": channels, "regions": regions, "statistics": channels * 8,
                "windows": config["windows"], "shrinkage": config["shrinkage"],
                "widths": widths, "primary_input": "measured Frank XYZ only",
                "context": "Offline full-record wavelets, then original cached R-peak crops; not a causal streaming model",
                "region_disclosure": "R-relative approximate regions, not validated QRS/ST/T delineations"}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return destination


def _fit_arm(arm, X, y, groups, rank, model_params, config, manifest, seed):
    weights = patient_balanced_weights(y, groups)
    projection = None
    if arm.endswith("_shared"):
        projection = SharedRelationalProjection(rank=rank, channels=manifest["channels"],
            regions=manifest["regions"], statistics=manifest["statistics"], covariance=arm == "covariance_shared")
        projection.fit(X, sample_weight=weights)
        X = projection.transform(X)
    estimator = make_estimator("xgboost", model_params, seed, len(config["classes"]))
    estimator.set_params(n_jobs=int(config.get("model_n_jobs", 4)))
    estimator.fit(X, y, sample_weight=weights)
    return projection, estimator


def _predict_arm(projection, estimator, X, n_classes):
    return predict_probabilities(estimator, X if projection is None else projection.transform(X), n_classes)


def run_relational_study(config: dict, arms: list[str] | None = None,
                         outer_folds: list[int] | None = None) -> pd.DataFrame:
    selected = list(ARMS) if arms is None else arms
    if not selected or set(selected) - set(ARMS):
        raise ValueError(f"Unknown or empty arms; choose from {ARMS}")
    bank = Path(config["bank_dir"])
    manifest = json.loads((bank / "manifest.json").read_text())
    groups = np.load(bank / "groups.npy", allow_pickle=False)
    labels = np.load(bank / "y.npy", allow_pickle=False)
    mapping = {name: i for i, name in enumerate(config["classes"])}
    y = np.array([mapping[str(label)] for label in labels])
    folds = load_or_create_folds(y, groups, config)
    all_fold_ids = set(range(1, int(config["outer_splits"]) + 1))
    if set(folds.fold) != all_fold_ids:
        raise ValueError("Frozen fold IDs do not match configured outer_splits")
    chosen_folds = sorted(all_fold_ids if outer_folds is None else set(outer_folds))
    if not chosen_folds or set(chosen_folds) - all_fold_ids:
        raise ValueError("Requested outer folds are invalid")
    partial = set(chosen_folds) != all_fold_ids
    output = Path(config["output_dir"]) / ("partial_folds_" + "_".join(map(str, chosen_folds)) if partial else "complete")
    output.mkdir(parents=True, exist_ok=True)
    if config.get("selection_metric", "patient_macro_f1") != "patient_macro_f1" or config.get("aggregation", "mean_probability") != "mean_probability":
        raise ValueError("This study supports patient macro F1 selection and mean-probability aggregation only")
    study_hash = hashlib.sha256(json.dumps({"config": config, "bank": manifest["fingerprint"]}, sort_keys=True).encode()).hexdigest()
    lookup = dict(zip(folds.patient_id.astype(str), folds.fold, strict=True))
    beat_folds = np.array([lookup[str(group)] for group in groups])
    statistics = np.load(bank / "statistics_context.npy", allow_pickle=False)
    summary = []
    for arm in selected:
        if arm.startswith("statistics_"):
            X = np.load(bank / f"{arm}.npy", allow_pickle=False)
        else:
            source = "covariance" if arm.startswith("covariance_") else "correlation"
            X = np.column_stack([np.load(bank / f"{source}.npy", mmap_mode="r", allow_pickle=False), statistics])
        probabilities = np.full((len(y), len(mapping)), np.nan)
        audits = []
        for fold in chosen_folds:
            checkpoint = output / f"fold_{fold:02d}_{arm}.npz"
            audit_path = output / f"fold_{fold:02d}_{arm}.json"
            train, test = np.flatnonzero(beat_folds != fold), np.flatnonzero(beat_folds == fold)
            if set(groups[train]) & set(groups[test]):
                raise RuntimeError("Outer patient leakage")
            if checkpoint.exists() and audit_path.exists():
                audit = json.loads(audit_path.read_text())
                if audit["study_fingerprint"] != study_hash:
                    raise ValueError("Study settings changed; select a new output_dir")
                with np.load(checkpoint, allow_pickle=False) as saved:
                    if not np.array_equal(saved["indices"], test):
                        raise ValueError("Checkpoint test rows changed")
                    probabilities[test] = saved["probabilities"]
                audits.append(audit)
                print(f"{arm} fold {fold}: resumed completed checkpoint", flush=True)
                continue
            inner = list(inner_patient_splits(y[train], groups[train], int(config["inner_splits"]), int(config["seed"]) + fold * 1000))
            candidates = []
            ranks = config["ranks"] if arm.endswith("_shared") else [None]
            combinations = [(rank, params) for rank in ranks for params in config["model_candidates"]]
            for rank, params in combinations:
                scores = []
                # Fixed candidates require no selection. Otherwise ALL learned
                # projection/scaling parameters are refitted in each inner split.
                if len(combinations) > 1:
                    for inner_index, (itrain, ivalid) in enumerate(inner):
                        ti, vi = train[itrain], train[ivalid]
                        if set(groups[ti]) & set(groups[vi]):
                            raise RuntimeError("Inner patient leakage")
                        projection, estimator = _fit_arm(arm, X[ti], y[ti], groups[ti], rank, params, config, manifest,
                                                       int(config["seed"]) + fold * 1000 + inner_index)
                        pred = _predict_arm(projection, estimator, X[vi], len(mapping))
                        _, truth, patient_scores = aggregate_patients(pred, y[vi], groups[vi])
                        scores.append(float(f1_score(truth, patient_scores.argmax(1), labels=np.arange(len(mapping)), average="macro", zero_division=0)))
                candidates.append({"rank": rank, "model_params": params, "inner_scores": scores,
                                   "inner_mean_macro_f1": float(np.mean(scores)) if scores else None})
            best = max(candidates, key=lambda row: row["inner_mean_macro_f1"] if row["inner_mean_macro_f1"] is not None else 0)
            projection, estimator = _fit_arm(arm, X[train], y[train], groups[train], best["rank"], best["model_params"], config, manifest, int(config["seed"]) + fold)
            probabilities[test] = _predict_arm(projection, estimator, X[test], len(mapping))
            _, truth, patient_scores = aggregate_patients(probabilities[test], y[test], groups[test])
            audit = {"study_fingerprint": study_hash, "fold": fold, "arm": arm,
                     "train_patients": len(np.unique(groups[train])), "test_patients": len(np.unique(groups[test])),
                     "train_beats": len(train), "test_beats": len(test), "patient_overlap": 0,
                     "best_candidate": best, "inner_candidates": candidates,
                     "projection_fit_scope": "inner training only during selection; outer training only for final fold model",
                     "projection_energy_fraction": projection.energy_fraction_ if projection is not None else None,
                     "output_feature_count": int(estimator.n_features_in_),
                     "patient_metrics": metric_bundle(truth, patient_scores, list(mapping))}
            np.savez_compressed(checkpoint, indices=test, probabilities=probabilities[test])
            audit_path.write_text(json.dumps(audit, indent=2))
            audits.append(audit)
            print(f"{arm} fold {fold}/{config['outer_splits']}: patient macro F1={audit['patient_metrics']['macro_f1']:.4f}", flush=True)
        evaluated = np.isfinite(probabilities).all(axis=1)
        if not partial and not evaluated.all():
            raise RuntimeError("Complete study is missing out-of-fold predictions")
        ids, truth, patient_scores = aggregate_patients(probabilities[evaluated], y[evaluated], groups[evaluated])
        result = {"arm": arm, "classes": list(mapping), "study_fingerprint": study_hash,
                  "feature_bank_fingerprint": manifest["fingerprint"], "complete": not partial,
                  "evaluated_folds": chosen_folds, "n_patients": len(ids), "n_beats": int(evaluated.sum()),
                  "protocol": "VCG-only development study on previously inspected frozen PTB folds; nested rank/model selection when candidates differ",
                  "weighting": "equal classes and equal patients within class", "aggregation": "mean beat probability per patient",
                  "patient_metrics": metric_bundle(truth, patient_scores, list(mapping)), "folds": audits}
        (output / f"metrics_{arm}.json").write_text(json.dumps(result, indent=2))
        predictions = pd.DataFrame({"patient_id": ids, "truth": [config["classes"][i] for i in truth],
                                    "prediction": [config["classes"][i] for i in patient_scores.argmax(1)],
                                    "fold": [lookup[str(pid)] for pid in ids]})
        for i, name in enumerate(config["classes"]):
            predictions[f"probability_{name}"] = patient_scores[:, i]
        predictions.to_csv(output / f"patient_predictions_{arm}.csv", index=False)
        save_confusion(truth, patient_scores.argmax(1), config["classes"], arm, output)
        summary.append({"arm": arm, "complete": not partial, "n_patients": len(ids),
                        "patient_macro_f1": result["patient_metrics"]["macro_f1"],
                        "patient_accuracy": result["patient_metrics"]["accuracy"]})
        pd.DataFrame(summary).to_csv(output / "comparison.csv", index=False)
    return pd.DataFrame(summary)
