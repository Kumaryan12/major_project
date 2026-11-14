"""Recording-overlap screening; deliberately not a patient-identity clearance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.fft import rfft, irfft, next_fast_len
from scipy.signal import resample_poly
import wfdb
import yaml
from joblib import Parallel, delayed

from . import external, signal, pretraining_source
from .external import ECG_LEADS, kors_transform
from .neural import _hash, _save_json
from .regularization import load_bank
from .signal_audit import file_hash


def templates(ecg, config):
    rate, points = config["screen_rate"], config["template_points"]
    before, after = int(np.ceil(0.25 * rate)), int(np.ceil(0.4 * rate))
    peaks = signal.detect_r_peaks(kors_transform(ecg, list(ECG_LEADS)), rate, 250, 150, 0.06)
    peaks = peaks[(peaks >= before) & (peaks + after < len(ecg))]
    if not len(peaks):
        return np.zeros((4, 8 * points), dtype=np.float32), False
    beats = np.stack([ecg[peak - before:peak + after + 1] for peak in peaks])
    chosen = [np.median(beats, 0), beats[0], beats[len(beats) // 2], beats[-1]]
    grid = np.linspace(-0.25 * rate, 0.4 * rate, points)
    output = []
    for beat in chosen:
        values = np.stack([np.interp(grid, np.arange(-before, after + 1), beat[:, lead]) for lead in range(8)])
        values -= values.mean(1, keepdims=True)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values = values / np.maximum(norms, 1e-12)
        flattened = values.ravel()
        output.append(flattened / max(np.linalg.norm(flattened), 1e-12))
    return np.asarray(output, dtype=np.float32), True


class CorrelationTarget:
    """Reuse target FFT and rolling variances for many independent queries."""
    def __init__(self, values, query_length):
        self.values = np.asarray(values, dtype=np.float64)
        self.length = query_length
        if self.values.ndim != 2 or self.values.shape[1] != 8 or len(values) < query_length or not np.isfinite(values).all():
            raise ValueError("Expected finite common-lead ECG longer than the query")
        self.nfft = next_fast_len(len(values) + query_length - 1)
        self.fft = rfft(self.values, self.nfft, axis=0)
        cumulative = np.vstack([np.zeros((1, 8)), np.cumsum(self.values, axis=0)])
        squares = np.vstack([np.zeros((1, 8)), np.cumsum(self.values ** 2, axis=0)])
        sums = cumulative[query_length:] - cumulative[:-query_length]
        self.variance = np.maximum(squares[query_length:] - squares[:-query_length] - sums ** 2 / query_length, 0)

    def compare(self, source, config):
        source = np.asarray(source, dtype=float)
        if source.ndim != 2 or source.shape[1] != 8 or len(source) < self.length or not np.isfinite(source).all():
            raise ValueError("Invalid source query ECG")
        offset = (len(source) - self.length) // 2
        query = source[offset:offset + self.length]
        centered = query - query.mean(0)
        norm2 = np.sum(centered ** 2, axis=0)
        active = np.sqrt(norm2 / self.length) >= config["minimum_lead_sd_mv"]
        count = int(active.sum())
        empty = {"candidate": False, "active_leads": count, "query_mean_correlation": 0.0, "query_minimum_correlation": 0.0,
                 "target_query_start": None, "full_source_available": False, "full_mean_correlation": None,
                 "full_minimum_correlation": None, "exact_full_segment_equal": False}
        if count < config["minimum_active_leads"]:
            return empty
        numerator = irfft(self.fft * np.conj(rfft(centered, self.nfft, axis=0)), self.nfft, axis=0)[:len(self.values) - self.length + 1]
        denominator = np.sqrt(self.variance[:, active] * norm2[active][None])
        correlations = np.divide(numerator[:, active], denominator, out=np.zeros_like(denominator), where=denominator > 1e-18)
        correlations = np.clip(correlations, -1, 1)
        means, minima = correlations.mean(1), correlations.min(1)
        eligible = (means >= config["mean_correlation_threshold"]) & (minima >= config["minimum_lead_correlation"])
        positions = np.flatnonzero(eligible)
        index = int(positions[np.argmax(means[positions])]) if len(positions) else int(np.argmax(means))
        start = index - offset
        full = start >= 0 and start + len(source) <= len(self.values)
        result = dict(empty, candidate=bool(len(positions)), query_mean_correlation=float(means[index]),
                      query_minimum_correlation=float(minima[index]), target_query_start=index, full_source_available=bool(full))
        if full:
            target = self.values[start:start + len(source)]
            a, b = source - source.mean(0), target - target.mean(0)
            r = np.sum(a[:, active] * b[:, active], 0) / np.maximum(np.linalg.norm(a[:, active], axis=0) * np.linalg.norm(b[:, active], axis=0), 1e-18)
            result.update(full_mean_correlation=float(np.mean(r)), full_minimum_correlation=float(np.min(r)),
                          exact_full_segment_equal=bool(np.array_equal(source, target)))
        return result


def rank_candidates(source_features, target_features, source_available, target_available, count):
    scores = source_features.reshape(-1, source_features.shape[-1]) @ target_features.reshape(-1, target_features.shape[-1]).T
    scores = scores.reshape(len(source_features), 4, len(target_features), 4).max((1, 3))
    missing_targets = np.flatnonzero(~np.asarray(target_available))
    return [np.arange(len(target_features)) if not source_available[index] else np.unique(np.concatenate([
        np.argsort(-scores[index], kind="stable")[:min(count, len(target_features))], missing_targets])) for index in range(len(source_features))]


def _cache_record(item, config, fingerprint):
    cache = Path(config["cache_dir"])
    name = hashlib.sha256(item["id"].encode()).hexdigest()[:24]
    path = cache / f"{name}.npz"
    if path.exists():
        with np.load(path) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            ecg, features = saved["ecg"], saved["features"]
            digest = hashlib.sha256(ecg.tobytes() + features.tobytes()).hexdigest()
            if metadata["fingerprint"] != fingerprint or metadata["digest"] != digest or features.shape != (4, 8 * config["template_points"]):
                raise ValueError("Overlap cache inputs/content changed")
            return str(path), features, bool(metadata["templates_available"])
    header = wfdb.rdheader(item["path"])
    names = [value.lower() for value in header.sig_name]
    if any(names.count(lead) != 1 for lead in ECG_LEADS):
        raise ValueError("Missing/duplicate common ECG leads for overlap screening")
    record = wfdb.rdrecord(item["path"], channels=[names.index(lead) for lead in ECG_LEADS])
    if record.fs != item["sampling_rate"] or any(str(unit).lower() != "mv" for unit in record.units) or not np.isfinite(record.p_signal).all():
        raise ValueError("Overlap screening requires finite common ECG leads in mV at the declared rate")
    ratio = int(record.fs) // config["screen_rate"]
    if ratio * config["screen_rate"] != record.fs:
        raise ValueError("Non-integral overlap resampling rate")
    ecg = resample_poly(record.p_signal, 1, ratio, axis=0).astype(np.float32)
    features, available = templates(ecg, config)
    metadata = {"fingerprint": fingerprint, "digest": hashlib.sha256(ecg.tobytes() + features.tobytes()).hexdigest(), "templates_available": available}
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, ecg=ecg, features=features, metadata=json.dumps(metadata))
    temporary.replace(path)
    return str(path), features, available


def _scan_target(target_index, source_indices, target_cache, source_cache, config):
    with np.load(target_cache[target_index]) as saved:
        target = CorrelationTarget(saved["ecg"], int(config["query_seconds"] * config["screen_rate"]))
    rows = []
    for index in source_indices:
        with np.load(source_cache[index]) as saved:
            result = target.compare(saved["ecg"], config)
        rows.append({"source_index": int(index), "target_index": int(target_index), **result})
    return rows


def _controls(target_cache, target_features, target_available, config):
    rng = np.random.default_rng(config["seed"])
    chosen = rng.choice(len(target_cache), min(config["positive_controls"], len(target_cache)), replace=False)
    controls, features, available = [], [], []
    for index in chosen:
        with np.load(target_cache[index]) as saved:
            target = saved["ecg"]
        length = config["screen_rate"] * 10
        if len(target) < length:
            raise ValueError("Target record is too short for the positive-control copy")
        start = int(rng.integers(0, len(target) - length + 1))
        copied = target[start:start + length].astype(float) * 1.03 + np.linspace(-0.02, 0.02, 8)[None]
        copied += rng.normal(0, 0.0001, copied.shape)
        feature, present = templates(copied, config)
        controls.append((int(index), copied))
        features.append(feature)
        available.append(present)
    for count in config["candidate_counts"]:
        candidates = rank_candidates(np.stack(features), target_features, available, target_available, count)
        rows = []
        for (index, copied), retrieved in zip(controls, candidates):
            with np.load(target_cache[index]) as saved:
                result = CorrelationTarget(saved["ecg"], int(config["query_seconds"] * config["screen_rate"])).compare(copied, config)
            rows.append({"known_target_index": index, "retrieved_known_record": bool(index in retrieved), "confirmed_copied_segment": bool(result["candidate"]),
                         "query_correlation": result["query_mean_correlation"]})
        if all(row["retrieved_known_record"] and row["confirmed_copied_segment"] for row in rows):
            return count, rows
    return None, rows


def validate_config(config):
    if config["screen_rate"] != 250 or config["query_seconds"] != 8 or config["template_points"] != 66:
        raise ValueError("Freeze the 250 Hz, eight-second multilead screening design")
    if not config["candidate_counts"] or config["candidate_counts"] != sorted(set(config["candidate_counts"])) or any(not isinstance(value, int) or value < 1 for value in config["candidate_counts"]):
        raise ValueError("Need increasing positive retrieval candidate counts")
    if not 0 < config["minimum_lead_correlation"] <= config["mean_correlation_threshold"] <= 1 or not 6 <= config["minimum_active_leads"] <= 8 or config["minimum_lead_sd_mv"] <= 0 or config["positive_controls"] < 1 or config["workers"] < 1:
        raise ValueError("Invalid overlap-screen thresholds/budget")


def run_screen(config):
    validate_config(config)
    source = yaml.safe_load(Path(config["source_config"]).read_text())
    target = yaml.safe_load(Path(config["target_config"]).read_text())
    source_bank = pretraining_source.prepare_bank(source)
    _, y, groups, target_bank = load_bank(target)
    cohort = pd.read_csv(Path(source["audit_dir"]) / "review/selected_cohort.csv")
    retained = pd.read_csv(Path(source["audit_dir"]) / "review/retained_records.csv")
    cohort = cohort.loc[cohort.ecg_id.isin(retained.ecg_id)].sort_values("ecg_id").reset_index(drop=True)
    paper = yaml.safe_load(Path(target["paper_config"]).read_text())
    beats = pd.read_csv(target["beats_path"])
    beats = beats.loc[beats.label.isin(target["classes"])]
    sources, targets = [], []
    for row in cohort.itertuples(index=False):
        pretraining_source.assert_source_row(row, source)
        sources.append({"id": f"ptbxl:ecg:{row.ecg_id}", "patient_id": row.patient_id, "path": str(Path(source["waveform_dir"]) / row.filename_hr), "sampling_rate": 500})
    for record, part in beats.groupby("record", sort=True):
        targets.append({"id": record, "patient_id": str(part.patient_id.iloc[0]), "path": str(Path(paper["data"]["raw_dir"]) / record), "sampling_rate": 1000})
    inventories = {role: [{**item, "header_sha256": file_hash(item["path"] + ".hea"), "data_sha256": file_hash(item["path"] + ".dat")} for item in items] for role, items in [("source", sources), ("target", targets)]}
    scope = {"config": config, "source_bank_manifest_sha256": file_hash(source_bank / "manifest.json"), "target_bank": target_bank,
        "source_inventory_sha256": _hash(inventories["source"]), "target_inventory_sha256": _hash(inventories["target"]),
        "code_sha256": file_hash(__file__), "signal_code_sha256": file_hash(signal.__file__), "kors_code_sha256": file_hash(external.__file__),
        "numpy": np.__version__, "scipy": scipy.__version__, "wfdb": wfdb.__version__}
    fingerprint = _hash(scope)
    output, cache = Path(config["output_dir"]), Path(config["cache_dir"])
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    guard = output / "screen_manifest.json"
    manifest = dict(scope, screen_fingerprint=fingerprint)
    if guard.exists() and json.loads(guard.read_text()) != manifest:
        raise ValueError("Overlap screen settings/data/code changed; choose new output/cache directories")
    _save_json(guard, manifest)
    if (output / "summary.json").exists():
        summary = json.loads((output / "summary.json").read_text())
        if summary["screen_fingerprint"] != fingerprint:
            raise ValueError("Overlap summary fingerprint changed")
        return output / "REPORT.md"
    review = output / "review"
    review.mkdir(exist_ok=True)
    _save_json(review / "input_provenance.json", inventories)
    cached = Parallel(n_jobs=config["workers"])(delayed(_cache_record)(item, config, fingerprint) for item in sources + targets)
    source_cache = [item[0] for item in cached[:len(sources)]]
    target_cache = [item[0] for item in cached[len(sources):]]
    source_features = np.stack([item[1] for item in cached[:len(sources)]])
    target_features = np.stack([item[1] for item in cached[len(sources):]])
    source_available = np.array([item[2] for item in cached[:len(sources)]])
    target_available = np.array([item[2] for item in cached[len(sources):]])
    count, controls = _controls(target_cache, target_features, target_available, config)
    pd.DataFrame(controls).to_csv(review / "positive_controls.csv", index=False)
    if count is None:
        raise ValueError("Known copied-segment controls failed retrieval/confirmation; target comparison must not start")
    candidates = rank_candidates(source_features, target_features, source_available, target_available, count)
    tasks = {index: [] for index in range(len(targets))}
    for source_index, retrieved in enumerate(candidates):
        for target_index in retrieved:
            tasks[int(target_index)].append(source_index)
    print(f"Screening {sum(map(len, tasks.values())):,} retrieved/fallback pairs after {len(controls)} positive controls; top-{count} retrieval", flush=True)
    jobs = Parallel(n_jobs=config["workers"], return_as="generator_unordered")(
        delayed(_scan_target)(index, indices, target_cache, source_cache, config) for index, indices in tasks.items() if indices)
    rows = []
    for number, values in enumerate(jobs, 1):
        rows.extend(values)
        if number % 25 == 0:
            print(f"Continuous ECG comparisons: {number} target records processed", flush=True)
    pairs = pd.DataFrame(rows)
    pairs["source_record"] = pairs.source_index.map({index: item["id"] for index, item in enumerate(sources)})
    pairs["target_record"] = pairs.target_index.map({index: item["id"] for index, item in enumerate(targets)})
    pairs.to_csv(review / "continuous_pair_screen.csv", index=False)
    hits = pairs.loc[pairs.candidate]
    target_hashes = {item["data_sha256"] for item in inventories["target"]}
    exact_encoded = [item for item in inventories["source"] if item["data_sha256"] in target_hashes]
    incomplete = int((pairs.active_leads < config["minimum_active_leads"]).sum())
    summary = {"screen_fingerprint": fingerprint, "source_records": len(sources), "source_patients": int(cohort.patient_id.nunique()),
        "target_records": len(targets), "target_patients": int(beats.patient_id.nunique()), "retrieval_candidate_count": count,
        "continuous_pairs_checked": len(pairs), "all_possible_pairs": len(sources) * len(targets),
        "positive_controls": len(controls), "positive_controls_retrieved_and_confirmed": sum(row["retrieved_known_record"] and row["confirmed_copied_segment"] for row in controls),
        "template_unavailable_source_records": int((~source_available).sum()), "template_unavailable_target_records": int((~target_available).sum()),
        "incomplete_active_lead_pairs": incomplete, "exact_encoded_data_matches": len(exact_encoded),
        "near_recording_candidate_pairs": len(hits), "candidate_source_records": int(hits.source_index.nunique()),
        "maximum_checked_query_mean_correlation": float(pairs.query_mean_correlation.max()),
        "recording_screen_passed": bool(not len(exact_encoded) and hits.empty and not incomplete),
        "patient_identity_independence_cleared": False,
        "claim_scope": "exploratory transfer only; no patient-identity clearance or independent/SOTA confirmation",
        "limits": "Top-k morphology retrieval plus eight-second common-ECG correlation, with synthetic copy controls. Not an exhaustive all-lag all-pair search, near-identity recognition, partial-copy search under eight seconds or proof that the same people never recur."}
    _save_json(output / "summary.json", summary)
    lines = ["# Source–target recording-overlap screen", "", f"{len(sources):,} retained PTB-XL source records (training and source validation) versus {len(targets)} PTB records from {summary['target_patients']} target patients.", "",
        f"Exact encoded data-file matches: **{len(exact_encoded)}**. High continuous-correlation candidate pairs: **{len(hits)}**. Positive copied-segment controls retrieved and confirmed: **{summary['positive_controls_retrieved_and_confirmed']}/{len(controls)}**.", "",
        f"Checked {len(pairs):,} retrieved/fallback pairs out of {summary['all_possible_pairs']:,} possible record pairs. Candidate retrieval uses median/first/middle/last common-ECG beat templates; predeclared top-k escalation selects {count} after testing synthetic copied controls. Missing templates use exhaustive candidate fallback. At least six active leads are required. Incomplete active-lead pairs: {incomplete}.", "",
        "The eight independent ECG leads are read in common physical mV units and resampled to 250 Hz. Continuous FFT cross-correlation searches all valid lags of the source's central eight seconds in each candidate PTB recording. A candidate requires mean lead correlation ≥0.995 and every active lead ≥0.98. Full source-segment correlation/equality is recorded where its alignment fits. Positive controls are ten-second PTB copies with known gain, offset and small noise, used only to test the screen, never for pretraining or classification.", "",
        "## Interpretation and gate", "",
        f"Recording screen passed under this bounded design: **{summary['recording_screen_passed']}**. A detected match or incomplete screen stops the target experiment pending source review/quarantine and, if needed, source retraining. No target patient is removed or relabeled.", "",
        "Passing does not identify people or establish that source and target patients are disjoint. Similar beats are not duplicate evidence by themselves; only continuous-segment confirmations are flagged. The top-k design and synthetic controls do not measure population sensitivity, and shorter/modified copies or different recordings of the same person can be missed. Patient identity remains unresolved even if there are zero candidates.", "",
        "Any subsequent scratch-versus-pretrained comparison remains explicitly exploratory development on repeatedly inspected PTB folds, with the unresolved source-target identity limitation. It cannot be presented as clean patient-independent confirmation or SOTA. Existing source pretraining declarations remain unchanged.", "",
        "Only aggregate results/provenance are public. Patient/record candidate pairs, controls, raw signals and per-record caches remain local in `review/` and ignored data caches. No PTB classification predictions or error flags enter the screen, and no fold-10 PTB-XL waveforms are accessed.", "",
        "Method reference: [SciPy cross-correlation documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html).", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    return output / "REPORT.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/source_target_overlap.yaml")
    args = parser.parse_args()
    print(run_screen(yaml.safe_load(Path(args.config).read_text())))
