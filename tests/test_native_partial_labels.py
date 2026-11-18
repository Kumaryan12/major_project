import numpy as np
import torch

from mi_localization.native_partial_labels import make_plans, positive_loss, shuffle_mentions
from mi_localization.neural import _seed


def test_unmentioned_source_regions_contribute_zero_gradient():
    _seed(42, {'device': 'cpu', 'torch_threads': 2})
    logits = torch.tensor([[1., 5., -1.], [0., -3., -2.]], requires_grad=True)
    observed = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
    loss = positive_loss(logits, observed)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.all(logits.grad[observed == 0] == 0)
    assert torch.all(logits.grad[observed == 1] < 0)


def test_shuffle_preserves_tag_counts_and_minimizes_identical_vectors_without_self_pairs():
    codes = np.array([[0, 1, 1, 0]] * 8 + [[0, 0, 1, 0]] * 3 + [[0, 1, 0, 0], [1, 0, 1, 1]])
    partners = shuffle_mentions(codes, 42)
    np.testing.assert_array_equal(np.sort(partners), np.arange(13))
    assert np.all(partners != np.arange(13))
    np.testing.assert_array_equal(codes[partners].sum(axis=0), codes.sum(axis=0))
    assert np.count_nonzero(np.all(codes == codes[partners], axis=1)) == 3


def test_sample_plans_only_use_training_target_rows_and_balance_source_patients():
    config = dict(samples_per_epoch=8, batch_size=4, target_epochs=2, source_beats_per_patient_per_update=2)
    train = np.array([2, 8, 14, 15])
    pools = [np.array([0, 1]), np.array([2, 3, 4]), np.array([5, 6])]
    weights = np.array([.4, .2, .1, .3])
    target, source = make_plans(config, train, weights, pools, 42)
    assert target.shape == (2, 2, 4) and source.shape == (2, 2, 6)
    assert np.isin(target, train).all()
    for i, pool in enumerate(pools):
        assert np.isin(source[..., 2 * i:2 * i + 2], pool).all()
    again = make_plans(config, train, weights, pools, 42)
    np.testing.assert_array_equal(target, again[0])
    np.testing.assert_array_equal(source, again[1])
