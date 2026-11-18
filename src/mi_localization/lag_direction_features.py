"""Time reversal invariants and signed covariance of fixed measured XYZ windows."""
from pathlib import Path

import numpy as np
import yaml

from . import chronology_bank, regularization
from .multiview_bank import array_hash
from .signal_audit import file_hash

SYMMETRIC = np.triu_indices(3)
ANTISYMMETRIC = np.triu_indices(3, k=1)


def window_blocks(waveforms, windows, lags):
    """Return [beat, window, lag, 6 symmetric / 3 antisymmetric entries]."""
    beats = np.asarray(waveforms, dtype=np.float64)
    if beats.ndim != 3 or beats.shape[1] != 3 or not np.isfinite(beats).all():
        raise ValueError('Expected finite XYZ beats')
    sym, anti = [], []
    for start, stop in windows:
        if not 0 <= start < stop <= beats.shape[-1]:
            raise ValueError('Invalid window support')
        x = beats[:, :, start:stop]
        x = x - x.mean(axis=2, keepdims=True)
        s_row, a_row = [], []
        for lag in lags:
            if not 0 < lag < stop - start:
                raise ValueError('Every positive lag requires pairs inside its window')
            product = (x[:, :, :-lag] @ x[:, :, lag:].transpose(0, 2, 1)) / (x.shape[2] - lag)
            transpose = product.transpose(0, 2, 1)
            s = (product + transpose) * .5
            a = (product - transpose) * .5
            s_row.append(s[:, SYMMETRIC[0], SYMMETRIC[1]])
            a_row.append(a[:, ANTISYMMETRIC[0], ANTISYMMETRIC[1]])
        sym.append(np.stack(s_row, axis=1))
        anti.append(np.stack(a_row, axis=1))
    return np.stack(sym, axis=1), np.stack(anti, axis=1)


def balanced_signs(count, generator):
    signs = np.ones(count, dtype=np.int8)
    negatives = count // 2 + (int(generator.integers(2)) if count % 2 else 0)
    signs[generator.permutation(count)[:negatives]] = -1
    return signs


def scaled_block(magnitude, arms, train, base_dimensions, config):
    train_magnitude = magnitude[train]
    rms = np.sqrt(np.mean(train_magnitude**2, axis=0))
    scale = np.maximum(np.std(train_magnitude, axis=0),
                       np.maximum(config['scale_relative_floor'] * rms, config['scale_absolute_floor']))
    gain = np.sqrt(config['auxiliary_weight'] * base_dimensions / magnitude.shape[1])
    return {arm: values / scale * gain for arm, values in arms.items()}, scale, gain


def load_records(config, bank):
    convolution_source = bank.source_scope['source']['source_scope']
    inherited = convolution_source['bank']['bank_manifest']['scope']['source']
    original_path = bank.source_config['source_config']
    original = yaml.safe_load(Path(original_path).read_text())
    waves, labels, groups, scope = regularization.load_bank(original)
    path = Path(original['sequence_bank_dir']) / 'metadata.npz'
    if scope != inherited['source'] or file_hash(path) != inherited['metadata_sha256']:
        raise ValueError('Original measured beats or chronological rows changed')
    paper = yaml.safe_load(Path(original['paper_config']).read_text())
    if paper['data']['before_r'] != config['r_index'] or paper['data']['sampling_rate'] != 1000:
        raise ValueError('Require original 1,000 Hz R-relative crops')
    with np.load(path, allow_pickle=False) as f:
        records = chronology_bank.RecordBank(waves, labels, groups, provenance=inherited,
                                             **{key: f[key] for key in f.files})
    if (not np.array_equal(records.ids, bank.ids) or not np.array_equal(records.labels, bank.labels)
            or (len(waves), len(records.record_patients)) != (bank.n_beats, bank.n_records)):
        raise ValueError('The new tensor and stored baseline must have the exact same cohort')
    return records


def reduce_records(config, records, allowed=None):
    windows = [tuple(config['r_index'] + t for t in bounds) for bounds in config['windows_ms']]
    lags = config['lags_ms']
    seeds = config['projection_seeds']
    rows = {name: [] for name in ('magnitude', 'signed', *(str(seed) for seed in seeds))}
    sign_audit = {seed: [] for seed in seeds}
    record_indices = np.arange(len(records.record_patients)) if allowed is None else records.records_for(allowed)
    owners = records.record_patients[record_indices]
    n_beats = 0
    for i in record_indices:
        selected = records.rows(i)
        n_beats += len(selected)
        s, a = window_blocks(records.waves[selected], windows, lags)
        common = s.mean(axis=0)
        rows['magnitude'].append(np.concatenate((common, np.abs(a).mean(axis=0)), axis=-1).ravel())
        rows['signed'].append(np.concatenate((common, a.mean(axis=0)), axis=-1).ravel())
        for seed in seeds:
            generator = np.random.default_rng(np.random.SeedSequence([seed, config['sign_seed_offset'], int(i)]))
            signs = balanced_signs(len(selected), generator)
            if abs(int(signs.sum())) > 1:
                raise ValueError('The sign control must be balanced within a recording')
            sign_audit[seed].append(array_hash(signs))
            scrambled = (a * signs[:, None, None, None]).mean(axis=0)
            rows[str(seed)].append(np.concatenate((common, scrambled), axis=-1).ravel())
    arrays = {arm: np.stack(values) for arm, values in rows.items()}
    col_count = len(windows) * len(lags) * 9
    if any(x.shape != (len(record_indices), col_count) or not np.isfinite(x).all()
           for x in arrays.values()):
        raise ValueError('Missing or nonfinite record tensor')
    checks = {
        'record_vectors': {name: array_hash(x) for name, x in arrays.items()},
        'sign_sequences': {str(seed): array_hash(np.asarray(digests)) for seed, digests in sign_audit.items()},
        'records': len(record_indices), 'beats': n_beats, 'auxiliary_dimensions': col_count,
    }
    patient = {}
    for arm, x in arrays.items():
        patient[arm] = np.zeros((len(records.ids), col_count))
        for row, pid in enumerate(records.ids):
            if np.any(owners == pid):
                patient[arm][row] = x[owners == pid].mean(axis=0)
    return patient, checks
