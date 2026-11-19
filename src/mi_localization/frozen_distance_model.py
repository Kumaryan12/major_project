"""Training-only shared-covariance and Euclidean patient distance classifiers."""
from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler


def class_statistics(z, labels, classes):
    z, labels = np.asarray(z, dtype=np.float64), np.asarray(labels)
    if (z.ndim != 2 or not z.shape[0] or not z.shape[1] or labels.shape != (len(z),)
            or not np.isfinite(z).all() or not np.array_equal(np.unique(labels), np.arange(classes))):
        raise ValueError('Finite fitting rows and every declared class are required')
    centroids, covariances = [], []
    for label in range(classes):
        rows = z[labels == label]
        center = rows.mean(axis=0)
        residual = rows - center
        centroids.append(center)
        covariances.append(residual.T @ residual / len(rows))
    covariance = np.mean(covariances, axis=0)
    return np.stack(centroids), (covariance + covariance.T) / 2


def fit_distance(x, labels, classes, shrinkage, trace_floor=1e-12, eigenvalue_floor=1e-6,
                 score_divisor=128.):
    if (not np.isfinite([shrinkage, trace_floor, eigenvalue_floor, score_divisor]).all()
            or not 0 < shrinkage <= 1 or min(trace_floor, eigenvalue_floor, score_divisor) <= 0):
        raise ValueError('Invalid fixed distance settings')
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid fitting features')
    scaler = StandardScaler().fit(x)
    centroids, covariance = class_statistics(scaler.transform(x), labels, classes)
    tau = max(float(np.trace(covariance) / x.shape[1]), trace_floor)
    matrix = (1 - shrinkage) * covariance / tau + shrinkage * np.eye(x.shape[1])
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2)
    precision = (eigenvectors / np.maximum(eigenvalues, eigenvalue_floor)) @ eigenvectors.T
    if not np.isfinite(precision).all():
        raise ValueError('Non-finite distance precision')
    return dict(scaler=scaler, centroids=centroids, covariance=covariance, precision=precision,
                tau=tau, shrinkage=float(shrinkage), classes=classes,
                trace_floor=trace_floor, eigenvalue_floor=eigenvalue_floor,
                score_divisor=score_divisor)


def predict_distance(model, x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != model['centroids'].shape[1] or not np.isfinite(x).all():
        raise ValueError('Invalid query features')
    residual = model['scaler'].transform(x)[:, None, :] - model['centroids'][None, :, :]
    distance = np.einsum('nki,ij,nkj->nk', residual, model['precision'], residual)
    if not np.isfinite(distance).all() or np.any(distance < -1e-8):
        raise ValueError('Invalid squared distance')
    return softmax(-np.maximum(distance, 0) / model['score_divisor'], axis=1)
