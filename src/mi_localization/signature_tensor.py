"""Exact truncated signatures of piecewise-linear (time, X, Y, Z) paths."""
from __future__ import annotations

from itertools import combinations, product

import numpy as np

CHANNELS = ("time", "X", "Y", "Z")
FAMILIES = ("signed_area", "endpoint_polynomial", "signature2", "signature3")


def concatenate(left, right):
    """Chen product, through degree three; any broadcastable batch dimensions."""
    a, b, c = left
    u, v, w = right
    return (a+u, b+v+a[..., :, None]*u[..., None, :],
            c+w+b[..., :, :, None]*u[..., None, None, :]+a[..., :, None, None]*v[..., None, :, :])


def signature(path):
    """Balanced Chen reduction of segment tensor exponentials in float64.

    Tensor axes follow chronological coordinate words: S[i,j] integrates
    coordinate i before coordinate j. There is no log-signature conversion,
    learned basis, closed-loop interpolation, spatial rotation or normalization.
    """
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 3 or path.shape[-1] != 4 or path.shape[1] < 2 or not np.isfinite(path).all():
        raise ValueError("Expected finite batch x samples>=2 x four-coordinate paths")
    d = np.diff(path, axis=1)
    values = (d, d[..., :, None]*d[..., None, :]/2,
              d[..., :, None, None]*d[..., None, :, None]*d[..., None, None, :]/6)
    while values[0].shape[1] > 1:
        length = values[0].shape[1]
        stop = length-length % 2
        combined = concatenate(tuple(x[:, :stop:2] for x in values), tuple(x[:, 1:stop:2] for x in values))
        values = tuple(np.concatenate([new, old[:, -1:]], axis=1) if length % 2 else new
                       for new, old in zip(combined, values, strict=True))
    result = tuple(x[:, 0] for x in values)
    if any(not np.isfinite(x).all() for x in result):
        raise ValueError("Nonfinite signature tensor")
    return result


def endpoint_tensors(delta):
    return (delta, delta[..., :, None]*delta[..., None, :]/2,
            delta[..., :, None, None]*delta[..., None, :, None]*delta[..., None, None, :]/6)


def validate(config, length):
    if config["sampling_rate"] <= 0 or not 0 <= config["r_index"] < length:
        raise ValueError("Invalid physical clock/R index")
    if not config["windows"] or any(not 0 <= start < stop <= length or stop-start < 2
                                      for start, stop in config["windows"].values()):
        raise ValueError("Invalid fixed trajectory windows")


def names(config):
    result = {key: [] for key in FAMILIES}
    words = {level: ["_".join(CHANNELS[i] for i in word) for word in product(range(4), repeat=level)]
             for level in (1, 2, 3)}
    for window in config["windows"]:
        offsets = [f"{window}/physical_mean/{axis}" for axis in CHANNELS[1:]]
        for family in FAMILIES:
            if family == "signed_area":
                base = [f"{window}/level1/{word}" for word in words[1]]
                base += [f"{window}/area/{CHANNELS[i]}_{CHANNELS[j]}" for i, j in combinations(range(4), 2)]
            else:
                maximum = 2 if family == "signature2" else 3
                base = [f"{window}/level{level}/{word}" for level in range(1, maximum+1) for word in words[level]]
            result[family] += base+offsets
    return result


def extract(waves, config):
    """Per-beat/trajectory descriptors; physical offsets remain parallel inputs."""
    waves = np.asarray(waves, dtype=np.float64)
    if waves.ndim != 3 or waves.shape[1] != 3 or not len(waves) or not np.isfinite(waves).all():
        raise ValueError("Expected finite nonempty batch x XYZ x time")
    validate(config, waves.shape[-1])
    values = {key: [] for key in FAMILIES}
    for start, stop in config["windows"].values():
        x = waves[..., start:stop].transpose(0, 2, 1)
        clock = (np.arange(start, stop, dtype=np.float64)-config["r_index"])/config["sampling_rate"]
        path = np.concatenate([np.broadcast_to(clock[None, :, None], (len(x), len(clock), 1)), x], axis=-1)
        s1, s2, s3 = signature(path)
        e1, e2, e3 = endpoint_tensors(path[:, -1]-path[:, 0])
        mean = x.mean(1)
        area = np.stack([.5*(s2[:, i, j]-s2[:, j, i]) for i, j in combinations(range(4), 2)], axis=-1)
        values["signed_area"].append(np.concatenate([s1, area, mean], axis=1))
        values["endpoint_polynomial"].append(np.concatenate([e1, e2.reshape(len(x), -1), e3.reshape(len(x), -1), mean], axis=1))
        values["signature2"].append(np.concatenate([s1, s2.reshape(len(x), -1), mean], axis=1))
        values["signature3"].append(np.concatenate([s1, s2.reshape(len(x), -1), s3.reshape(len(x), -1), mean], axis=1))
    result = {key: np.concatenate(value, axis=1) for key, value in values.items()}
    schema = names(config)
    if any(value.shape != (len(waves), len(schema[key])) or not np.isfinite(value).all() for key, value in result.items()):
        raise ValueError("Invalid signature descriptor schema/content")
    return result


def mathematical_audit(path):
    """Algebraic checks independent of class labels; diagnostic errors only."""
    s1, s2, s3 = signature(path)
    first = float(np.max(np.abs(s1-(path[:, -1]-path[:, 0]))))
    shuffle2 = float(np.max(np.abs(s1[:, :, None]*s1[:, None, :]-s2-s2.transpose(0, 2, 1))))
    shuffle3 = max(float(np.max(np.abs(s1[:, i]*s2[:, j, k]-s3[:, i, j, k]-s3[:, j, i, k]-s3[:, j, k, i])))
                   for i, j, k in product(range(4), repeat=3))
    return {"endpoint_max_abs_error": first, "degree2_shuffle_max_abs_error": shuffle2,
            "degree3_shuffle_max_abs_error": shuffle3}
