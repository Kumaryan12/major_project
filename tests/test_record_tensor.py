from pathlib import Path

import numpy as np
import pytest
import yaml

from mi_localization.record_tensor_audit import diagnose,prototype_views


def config():
    return yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/record_tensor_audit.yaml").read_text())


def test_views_retain_original_signal_and_physiological_variation():
    x=np.random.default_rng(42).normal(0,.2,(12,3,651)).astype(np.float32)
    saved=x.copy()
    views=prototype_views(x)
    assert views.shape==(12,3,3,651)
    np.testing.assert_array_equal(views[:,0],x)
    np.testing.assert_allclose(views[:,1]+views[:,2],x,atol=1e-7,rtol=1e-6)
    np.testing.assert_array_equal(x,saved)
    assert np.any(views[:,2]!=0)
    with pytest.raises(ValueError,match="finite"):
        prototype_views(np.full_like(x,np.nan))


def test_known_common_xyz_jitter_is_detected_without_mutating_the_source():
    cfg=config()
    t=np.arange(651)
    base=np.exp(-((t-250)/14)**2)[None,:]*np.array([[1.],[-.6],[.8]])
    x=np.repeat(base[None],8,axis=0)
    # Each odd/even reference half has three normal and one shifted beat.
    x[[0,1]]=np.roll(base,12,axis=1)
    saved=x.copy()
    summary,lags=diagnose(x,cfg)
    np.testing.assert_array_equal(lags[:2],[12,12])
    np.testing.assert_array_equal(lags[2:],np.zeros(6))
    assert summary["evaluated"] and summary["fraction_abs_lag_ge10ms"]==.25
    np.testing.assert_array_equal(x,saved)


def test_flat_trace_prefers_zero_shift_and_short_records_stay_explicit():
    summary,lags=diagnose(np.zeros((8,3,651)),config())
    assert summary["evaluated"] and np.isfinite(list(v for v in summary.values() if isinstance(v,float))).all()
    np.testing.assert_array_equal(lags,np.zeros(8))
    summary,lags=diagnose(np.zeros((3,3,651)),config())
    assert not summary["evaluated"] and summary["beats"]==3 and lags is None
