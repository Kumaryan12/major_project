import numpy as np
import torch
from torch.nn import functional as F

from mi_localization.benchmark import patient_balanced_weights
from mi_localization.neural import _seed
from mi_localization.patient_probability_loss import cross_entropies, sampling_plan


def test_losses_match_direct_probability_pooling_and_per_beat_cross_entropy():
    _seed(42, {'device': 'cpu', 'torch_threads': 2})
    logits = torch.tensor([[[1., 2., 3.], [4., 3., 2.]], [[-1., 0., 1.], [0., 3., -2.]]], requires_grad=True)
    targets = torch.tensor([2, 1])
    beat, pooled = cross_entropies(logits, targets)
    expected_beat = F.cross_entropy(logits.flatten(0, 1), targets.repeat_interleave(2))
    expected_pooled = F.nll_loss(logits.softmax(-1).mean(1).log(), targets)
    torch.testing.assert_close(beat, expected_beat)
    torch.testing.assert_close(pooled, expected_pooled)
    assert pooled < beat
    (0.5 * (beat + pooled)).backward()
    assert torch.isfinite(logits.grad).all()


def test_stable_pooling_with_tiny_probabilities_and_single_beat_bags():
    _seed(42, {'device': 'cpu', 'torch_threads': 2})
    logits = torch.tensor([[[1000., -1000.]], [[2000., -2000.]]], requires_grad=True)
    beat, pooled = cross_entropies(logits, torch.tensor([1, 1]))
    torch.testing.assert_close(beat, pooled)
    assert pooled.item() == 3000.
    pooled.backward()
    assert torch.isfinite(logits.grad).all()


def test_bags_are_reproducible_class_balanced_and_sample_training_patients_only():
    groups = np.array(['a'] * 4 + ['b'] * 16 + ['c'] * 2 + ['held_out'])
    labels = np.array([0] * 20 + [1] * 3)
    train = np.arange(len(groups) - 1)
    weights = patient_balanced_weights(labels[train], groups[train])
    settings = dict(beats_per_bag=8, batch_size=128, samples_per_epoch=2048, target_epochs=8)
    rows = sampling_plan(settings, train, labels, groups, weights, 42)
    assert rows.shape == (8, 16, 16, 8)
    np.testing.assert_array_equal(rows, sampling_plan(settings, train, labels, groups, weights, 42))
    patient_bags = groups[rows]
    assert np.all(patient_bags == patient_bags[..., :1])
    assert np.isin(rows, train).all()
    np.testing.assert_allclose(
        [(patient_bags[..., 0] == p).mean() for p in ('a', 'b', 'c')],
        [.25, .25, .5], atol=.035)
