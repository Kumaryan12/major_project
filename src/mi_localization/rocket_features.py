"""Bounded MultiRocket-style features and upstream experimental multivariate HYDRA.

MultiRocket pooling follows the paper's positive *thresholded* responses, not
the historical implementation's C+bias / gap-count conventions. This is an
explicit bounded adaptation, not a bitwise reproduction of that implementation.
HYDRA uses its unchanged GPL-3.0 upstream implementation; see _vendor/hydra.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
import torch
from torch.nn import functional as F
from scipy.fft import dct

from ._vendor.hydra.hydra_multivariate import HydraMultivariate
from .multiview_bank import array_hash

ARMS = ("dct", "multirocket", "hydra", "combined")


def check_waves(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 3 or x.shape[2] < 17 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Require finite recording × signed XYZ × time")
    return x


def pool_positive(z):
    """PPV, MPV, MIPV (zero-based; -1 empty), positive run length in samples."""
    positive = z > 0
    count = positive.sum(-1)
    denominator = count.clamp_min(1)
    index = torch.arange(z.shape[-1], device=z.device)
    last_nonpositive = torch.where(positive, -1, index).cummax(-1).values
    run = torch.where(positive, index-last_nonpositive, 0).amax(-1)
    ppv = count.to(z.dtype)/z.shape[-1]
    mpv = torch.where(positive, z, 0).sum(-1)/denominator
    mipv = torch.where(count > 0, (positive*index).sum(-1)/denominator, -1)
    return torch.stack([ppv, mpv, mipv, run], -1)


class MultiRocketFeatures:
    """84 length-nine kernels, one train-only quantile per dilation/kernel/view."""
    def __init__(self, length, dilations, seed):
        if not 1 <= dilations <= 32:
            raise ValueError("Invalid bounded dilation count")
        self.length, self.seed = length, seed
        self.dilations = [np.unique(np.logspace(0, np.log2((n-1)/8), dilations, base=2).astype(int)).tolist()
                          for n in (length, length-1)]
        self.weights, self.biases, self.bias_rows = [], [], []

    def fit(self, x, record_patients):
        x = check_waves(x)
        if x.shape[-1] != self.length or len(record_patients) != len(x):
            raise ValueError("MultiRocket fit shape mismatch")
        rng = np.random.default_rng(self.seed)
        record_patients = np.asarray(record_patients)
        ids = np.unique(record_patients)
        kernels = np.full((84, 9), -1, np.float32)
        for i, triple in enumerate(combinations(range(9), 3)):
            kernels[i, list(triple)] = 2
        self.weights, self.biases, self.bias_rows = [], [], []
        for view, ds in enumerate(self.dilations):
            values = x if view == 0 else np.diff(x, axis=-1)
            for di, dilation in enumerate(ds):
                masks = np.zeros((84, 3), np.float32)
                for mask in masks:
                    number = min(3, int(2**rng.uniform(0, 2)))
                    mask[rng.choice(3, number, replace=False)] = 1
                weight = kernels[:, None, :]*masks[:, :, None]
                # Each bias samples a patient uniformly, then one of their records.
                chosen = np.array([rng.choice(np.flatnonzero(record_patients == patient))
                                   for patient in rng.choice(ids, 84)])
                selected = torch.from_numpy(values[chosen].copy()).reshape(1, 84*3, -1)
                response = F.conv1d(selected, torch.from_numpy(weight), padding=4*dilation,
                                    dilation=dilation, groups=84)[0].numpy()
                quantiles = ((np.arange(di*84+1, (di+1)*84+1)*((5**.5+1)/2)) % 1)
                biases = np.array([np.quantile(r, q) for r, q in zip(response, quantiles)], np.float32)
                self.weights.append(weight)
                self.biases.append(biases)
                self.bias_rows.append(chosen)
        return self

    def transform(self, x, chunk_size=32):
        x = check_waves(x)
        if x.shape[-1] != self.length or len(self.weights) != sum(map(len, self.dilations)):
            raise ValueError("Unfitted or incompatible MultiRocket transform")
        chunks = []
        for start in range(0, len(x), chunk_size):
            raw = torch.from_numpy(x[start:start+chunk_size].copy())
            features, wi = [], 0
            for view, ds in enumerate(self.dilations):
                values = raw if view == 0 else torch.diff(raw)
                for di, dilation in enumerate(ds):
                    response = F.conv1d(values, torch.from_numpy(self.weights[wi]), padding=4*dilation, dilation=dilation)
                    response = response-torch.from_numpy(self.biases[wi])[None, :, None]
                    # Half the kernels use padding, half use only valid responses.
                    # Assemble in stable kernel order without mixing the lengths.
                    output = torch.empty((len(values), 84, 4))
                    for parity in (0, 1):
                        chosen = np.flatnonzero((np.arange(84)+di) % 2 == parity)
                        selected = response[:, chosen]
                        if parity: selected = selected[..., 4*dilation:-4*dilation]
                        output[:, chosen] = pool_positive(selected)
                    features.append(output.flatten(1).numpy())
                    wi += 1
            chunks.append(np.concatenate(features, axis=1))
        return np.concatenate(chunks)

    def audit(self):
        return {"seed": self.seed, "dilations": self.dilations,
                "weights": [array_hash(x) for x in self.weights],
                "biases": [array_hash(x) for x in self.biases],
                "bias_rows": [array_hash(x) for x in self.bias_rows]}


class FeatureTransform:
    """Shared training-only scalar RMS; all axes retain relative scale/orientation."""
    def __init__(self, config, seed):
        self.config, self.seed = config, seed

    def fit(self, templates, record_patients):
        x = check_waves(templates)
        ids = np.unique(record_patients)
        if len(record_patients) != len(x): raise ValueError("Template membership mismatch")
        energy = np.mean(x.astype(np.float64)**2, axis=(1, 2))
        self.scale = float(np.sqrt(np.mean([energy[np.asarray(record_patients) == p].mean() for p in ids])))
        if not np.isfinite(self.scale) or self.scale <= 1e-12: raise ValueError("Degenerate training RMS")
        self.multirocket = MultiRocketFeatures(x.shape[-1], self.config["multirocket_dilations"], self.seed)
        self.multirocket.fit(x/self.scale, record_patients)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.hydra = HydraMultivariate(x.shape[-1], 3, k=self.config["hydra_kernels"], g=self.config["hydra_groups"])
        self.training_patients, self.training_records = len(ids), len(x)
        return self

    def transform(self, templates):
        x = check_waves(templates)/self.scale
        values = {"dct": dct(x, type=2, axis=-1, norm="ortho")[..., :self.config["dct_coefficients"]].reshape(len(x), -1),
                  "multirocket": self.multirocket.transform(x, self.config["chunk_size"])}
        with torch.inference_mode():
            values["hydra"] = self.hydra.batch(torch.from_numpy(x.copy()), self.config["chunk_size"]).numpy()
        values["combined"] = np.concatenate([values["multirocket"], values["hydra"]], axis=1)
        if any(not np.isfinite(v).all() for v in values.values()): raise ValueError("Nonfinite convolutional features")
        return values

    def audit(self):
        return {"scale": self.scale, "training_patients": self.training_patients, "training_records": self.training_records,
                "multirocket": self.multirocket.audit(), "hydra_weights": [array_hash(x.numpy()) for x in self.hydra.W],
                "hydra_channels": [array_hash(x.numpy()) for x in self.hydra.I]}


def aggregate(features, record_patients, ids):
    """One row per patient, equal-record arithmetic mean; no labels involved."""
    record_patients, ids = np.asarray(record_patients), np.asarray(ids)
    if set(record_patients) != set(ids) or len(set(ids)) != len(ids):
        raise ValueError("Patient aggregation membership mismatch")
    return {name: np.stack([x[record_patients == p].mean(0) for p in ids]).astype(np.float64) for name, x in features.items()}
