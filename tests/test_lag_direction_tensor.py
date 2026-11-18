import numpy as np

from mi_localization.lag_direction_features import balanced_signs, scaled_block, window_blocks


def test_time_reversal_preserves_symmetric_part_and_flips_signed_part():
    t = np.linspace(0, 4 * np.pi, 100)
    x = np.stack((np.cos(t), 2 * np.sin(t), np.sin(2 * t)))[None]
    forward_sym, forward_anti = window_blocks(x, [(0, 100)], [4, 12])
    backward_sym, backward_anti = window_blocks(x[:, :, ::-1], [(0, 100)], [4, 12])
    np.testing.assert_allclose(forward_sym, backward_sym, atol=1e-12)
    np.testing.assert_allclose(forward_anti, -backward_anti, atol=1e-12)
    assert abs(forward_anti[0, 0, 0, 0]) > .1


def test_physical_amplitude_is_not_normalized_away():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(2, 3, 100))
    s, a = window_blocks(x, [(0, 100)], [4, 12])
    scaled_s, scaled_a = window_blocks(x * 2, [(0, 100)], [4, 12])
    np.testing.assert_allclose(scaled_s, 4 * s)
    np.testing.assert_allclose(scaled_a, 4 * a)


def test_sign_scramble_is_balanced_and_shared_scale_uses_training_only():
    generate = lambda: np.random.default_rng(24)
    even = balanced_signs(100, generate())
    odd = balanced_signs(101, generate())
    np.testing.assert_array_equal(even, balanced_signs(100, generate()))
    assert even.sum() == 0 and abs(odd.sum()) == 1
    np.testing.assert_array_equal(np.abs(odd), np.ones(101))
    settings = dict(auxiliary_weight=.25, scale_relative_floor=.1, scale_absolute_floor=1e-10)
    block = np.array([[2., 3.], [1., 5.], [4., 6.]])
    original, scale, gain = scaled_block(block, {'a': block}, np.array([0, 1]), 8, settings)
    changed = block.copy()
    changed[2] = 10000
    reloaded, scale2, gain2 = scaled_block(changed, {'a': changed}, np.array([0, 1]), 8, settings)
    np.testing.assert_array_equal(scale, scale2)
    np.testing.assert_array_equal(original['a'][:2], reloaded['a'][:2])
    assert gain == gain2
