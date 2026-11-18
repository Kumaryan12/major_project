from pathlib import Path

import numpy as np
import yaml

from mi_localization.local_timing_audit import (
    audit_record, delay_trace, fit_lags, lag_grid, record_controls,
)


def settings():
    config = yaml.safe_load(Path('configs/local_timing_audit_v1.yaml').read_text())
    grid = lag_grid(config)
    windows = [tuple(config['r_index'] + i for i in pair) for pair in config['windows'].values()]
    return config, grid, windows


def pulse_template():
    t = np.arange(651)
    near = 0.8 * np.exp(-0.5 * ((t - 250) / 10)**2)
    late = 0.3 * np.exp(-0.5 * ((t - 390) / 13)**2)
    direction = np.array([1., -0.8, 0.6])[:, None]
    return direction * near, direction * late


def test_joint_estimator_recovers_delay_with_amplitude_change_and_rejects_flat_trace():
    config, grid, windows = settings()
    early, late = pulse_template()
    reference = early + late
    delayed = 1.25 * delay_trace(reference, 12) + np.array([.1, -.1, .2])[:, None]
    beat_rows = np.stack((delayed, np.zeros_like(delayed)))
    fit = fit_lags(beat_rows, reference[None], windows[1], grid, config)
    assert grid[fit.lag_index[0, 0]] == 12
    assert fit.accepted.tolist() == [[True, False]]


def test_local_departure_finds_injected_post_r_shift_and_cancels_global_shift():
    config, grid, windows = settings()
    early, late = pulse_template()
    reference = early + late
    beats = np.stack([reference.copy() for _ in range(16)])
    beats[0] = early + delay_trace(late, 12)
    beats[4] = delay_trace(reference, 12)
    output = audit_record(beats, windows, grid, config)
    assert output['supported_local'][0, 1]
    assert output['mean_relative_lag_ms'][0, 1] == 12
    assert output['repeatable'][4, :2].all()
    assert output['mean_relative_lag_ms'][4, 1] == 0
    assert not output['supported_local'][4, 1]


def test_synthetic_control_reports_deterministic_recovery_in_informative_windows():
    config, grid, windows = settings()
    early, late = pulse_template()
    control = record_controls(early + late, windows, grid, config, np.random.default_rng(42))
    np.testing.assert_array_equal(control['estimated_delays'][:2],
                                  [config['control_delays_ms']] * 2)
    assert control['positive_correct'][:2].all()
    assert not control['null_apparent_shift'][:2].any()
