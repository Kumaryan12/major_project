import subprocess
import sys

import numpy as np
import pytest
from scipy.special import softmax

from mi_localization.frozen_distance_model import class_statistics, fit_distance, predict_distance


def test_identity_shrinkage_matches_analytic_centroid_distances():
    x = np.array([[-3., 0.], [-1., 2.], [1., 0.], [3., 2.]])
    labels = np.array([0, 0, 1, 1])
    head = fit_distance(x, labels, 2, 1., score_divisor=4.)
    query = np.array([[-2., 1.], [2., 1.]])
    z = (query - x.mean(0)) / x.std(0)
    centers = np.array([[-2., 1.], [2., 1.]])
    centers = (centers - x.mean(0)) / x.std(0)
    distances = ((z[:, None] - centers[None]) ** 2).sum(2)
    np.testing.assert_array_equal(head['precision'], np.eye(2))
    np.testing.assert_allclose(predict_distance(head, query), softmax(-distances / 4., axis=1), atol=1e-15)
    np.testing.assert_array_equal(predict_distance(head, query).argmax(1), [0, 1])


def test_covariance_gives_equal_mass_to_classes_and_is_not_total_covariance():
    z = np.array([[-4., 0.], [4., 0.], [9., -1.], [9., 1.]])
    labels = np.array([0, 0, 1, 1])
    centers, covariance = class_statistics(z, labels, 2)
    np.testing.assert_array_equal(centers, [[0., 0.], [9., 0.]])
    np.testing.assert_array_equal(covariance, [[8., 0.], [0., .5]])
    _, repeated = class_statistics(np.r_[z, z[:2], z[:2]], np.r_[labels, [0, 0, 0, 0]], 2)
    np.testing.assert_array_equal(covariance, repeated)


def test_shrinkage_is_positive_and_predictions_match_direct_quadratic_form():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(12, 5)); labels = np.repeat(np.arange(3), 4)
    head = fit_distance(x, labels, 3, .5)
    covariance = .5 * head['covariance'] / head['tau'] + .5 * np.eye(5)
    np.testing.assert_allclose(head['precision'], np.linalg.inv(covariance), atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() >= .5 - 1e-12
    query = rng.normal(size=(2, 5))
    delta = head['scaler'].transform(query)[:, None] - head['centroids'][None]
    distance = np.array([[row @ np.linalg.solve(covariance, row) for row in rows] for rows in delta])
    np.testing.assert_allclose(predict_distance(head, query), softmax(-distance / 128., axis=1))
    np.testing.assert_array_equal(predict_distance(head, query[::-1]), predict_distance(head, query)[::-1])
    # Query extremes are never included in the fitted scaler.
    np.testing.assert_array_equal(head['scaler'].mean_, x.mean(0))


def test_degenerate_features_stay_finite_and_invalid_training_is_rejected():
    head = fit_distance(np.ones((6, 4)), np.repeat(np.arange(3), 2), 3, .5)
    np.testing.assert_allclose(predict_distance(head, np.ones((2, 4))), np.full((2, 3), 1 / 3))
    for x, labels in [(np.zeros((4, 2)), np.zeros(4)), (np.full((6, 2), np.nan), np.repeat(np.arange(3), 2))]:
        with pytest.raises(ValueError):
            fit_distance(x, labels, 3, .9)


def test_distance_restore_query_isolation_and_corruption(tmp_path):
    code = r'''
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from mi_localization import neural
from mi_localization.frozen_distance_readout import distance_context

neural._seed(42, dict(device='cpu', torch_threads=2))
root=Path(sys.argv[1]);review=root/'review';review.mkdir()
context=review/'context.json';context.write_text('{}')
config=dict(classes=list('abcdef'),shrinkages=[1.,.9,.5],trace_floor=1e-12,
            eigenvalue_floor=1e-6,score_divisor=128.)
bank=SimpleNamespace(ids=np.array([str(i) for i in range(18)]),labels=np.tile(np.arange(6),3))
train=np.arange(12);query=np.arange(12,18);x=np.random.default_rng(7).normal(size=(18,64))
scope={'study_fingerprint':'synthetic'}
args=(bank,config,scope,review,42,1,1,'mahalanobis',1,train,query,x,context)
before={};p=distance_context(*args,False,before)
after={};restored=distance_context(*args,True,after)
np.testing.assert_array_equal(p,restored);assert before==after
bank.labels[query]=np.roll(bank.labels[query],1)
np.testing.assert_array_equal(p,distance_context(*args,True,{}))
bank.labels[train[0]]=1
try: distance_context(*args,True,{})
except ValueError: pass
else: raise AssertionError('Changed fitting labels must fail')
bank.labels[train[0]]=0
model_path=next(review.glob('head_*.joblib'))
model_path.write_bytes(model_path.read_bytes()+b'corrupted')
try: distance_context(*args,True,{})
except ValueError: pass
else: raise AssertionError('Corrupted model must fail')
'''
    subprocess.run([sys.executable, '-c', code, str(tmp_path)], check=True)
