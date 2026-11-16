"""Fixed signed morphology features and matched distribution-pooling controls."""
from __future__ import annotations

import numpy as np
from scipy.fft import dct
from scipy.special import ndtri

from .multiview_bank import array_hash

ARMS = ("median_waveform", "mean_embedding", "mean_spread", "quantiles")


def extract(waves, config):
    """One row per original beat, window, signed XYZ/DCT/physical descriptor."""
    x = np.asarray(waves, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Require finite nonempty measured XYZ waveforms")
    result = []
    for lo, hi in config["windows"]:
        if not 0 <= lo < hi <= x.shape[-1] or config["dct_coefficients"] > hi-lo:
            raise ValueError("Window/feature budget exceeds original crop")
        z = x[:, :, lo:hi]
        coefficients = dct(z, type=2, norm="ortho", axis=-1)[..., :config["dct_coefficients"]]/np.sqrt(hi-lo)
        physical = np.stack([np.sqrt(np.mean(z*z, axis=-1)), np.mean(np.abs(z), axis=-1),
                             z.min(axis=-1), z.max(axis=-1)], axis=-1)
        result.append(np.concatenate([coefficients, physical], axis=-1).reshape(len(x), -1))
    return np.stack(result, axis=1)


def summaries(projected, projected_median, quantiles):
    """The moment control occupies the same quantile grid using only mean/SD.

    No per-arm column standardization: repeated means are divided by sqrt(Q),
    preserving their Euclidean kernel rather than multiplying it Q times.
    """
    z = np.asarray(projected, dtype=np.float64)
    m = np.asarray(projected_median, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)
    if z.ndim != 3 or not len(z) or m.shape != z.shape[1:] or not np.isfinite(z).all() or not np.isfinite(m).all():
        raise ValueError("Invalid projected record")
    if len(q) < 3 or not np.isfinite(q).all() or np.any(q <= 0) or np.any(q >= 1) or np.any(np.diff(q) <= 0):
        raise ValueError("Require strictly increasing interior quantiles")
    mean = z.mean(axis=0)
    sd = np.sqrt(np.mean((z-mean)**2, axis=0))
    shape = (*m.shape, len(q))
    values = {
        "median_waveform": np.broadcast_to(m[..., None], shape),
        "mean_embedding": np.broadcast_to(mean[..., None], shape),
        "mean_spread": mean[..., None]+sd[..., None]*ndtri(q),
        "quantiles": np.moveaxis(np.quantile(z, q, axis=0, method="linear"), 0, -1),
    }
    return {a: np.asarray(v/np.sqrt(len(q))).reshape(-1) for a, v in values.items()}


class DistributionTransform:
    """Equal-patient/equal-record/equal-beat training-only morphology scaling."""

    def __init__(self, config, seed):
        self.config, self.seed = config, int(seed)

    def fit(self, bank, patients):
        patients = np.asarray(patients)
        if not len(patients) or len(np.unique(patients)) != len(patients) or not set(patients).issubset(set(bank.ids)):
            raise ValueError("Invalid training patients")
        records = bank.records_for(patients)
        # Two passes avoid cancellation/negative-variance roundoff.
        counts = {p: int(np.sum(bank.record_patients[records] == p)) for p in patients}
        mean = np.zeros(bank.features.shape[1:], dtype=np.float64)
        for record in records:
            weight = 1./(len(patients)*counts[bank.record_patients[record]])
            mean += weight*np.asarray(bank.features[bank.rows(record)]).mean(axis=0)
        variance = np.zeros_like(mean)
        for record in records:
            weight = 1./(len(patients)*counts[bank.record_patients[record]])
            variance += weight*np.mean((bank.features[bank.rows(record)]-mean)**2, axis=0)
        self.mean = mean
        self.scale = np.sqrt(variance)
        self.constant = self.scale <= self.config["feature_scale_floor"]
        self.scale[self.constant] = 1.
        rng = np.random.default_rng(self.seed)
        self.projection = rng.normal(size=(*mean.shape, self.config["projections"]))
        self.projection /= np.linalg.norm(self.projection, axis=1, keepdims=True)
        self.training_patients_sha256 = array_hash(np.sort(patients))
        if not np.isfinite(self.scale).all() or not np.isfinite(self.projection).all():
            raise ValueError("Invalid training-only feature scaling")
        return self

    def project(self, x):
        return np.einsum("bwd,wdp->bwp", (np.asarray(x)-self.mean)/self.scale, self.projection, optimize=True)

    def transform(self, bank, patients):
        patients = np.asarray(patients)
        records = bank.records_for(patients)
        values = {a: [] for a in ARMS}
        owner = []
        for record in records:
            z = self.project(bank.features[bank.rows(record)])
            median = self.project(bank.template_features[record:record+1])[0]
            pooled = summaries(z, median, self.config["quantiles"])
            for arm in ARMS: values[arm].append(pooled[arm])
            owner.append(bank.record_patients[record])
        owner = np.asarray(owner)
        # Recording mode is variable length, pooled equally per patient.
        result = {a: np.stack([np.mean(np.asarray(values[a])[owner == p], axis=0) for p in patients]) for a in ARMS}
        if any(not np.isfinite(x).all() for x in result.values()):
            raise ValueError("Nonfinite distribution features")
        return result

    def audit(self):
        return dict(seed=self.seed, training_patients_sha256=self.training_patients_sha256,
                    mean_sha256=array_hash(self.mean), scale_sha256=array_hash(self.scale),
                    projection_sha256=array_hash(self.projection), constant_columns=int(self.constant.sum()),
                    feature_dimensions=int(self.mean.shape[0]*self.config["projections"]*len(self.config["quantiles"])),
                    scaling="shared training-only morphology scaling; no per-arm standardization; quantile grid / sqrt(Q)")
