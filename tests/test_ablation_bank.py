from pathlib import Path

import numpy as np

from mi_localization.benchmark import load_primary_data


def test_benchmark_loads_named_feature_from_safe_archive(tmp_path: Path):
    path = tmp_path / "features.npz"
    np.savez_compressed(
        path,
        wavelet_statistics=np.arange(12, dtype=np.float32).reshape(3, 4),
        y=np.array(["AMI", "HC", "AMI"], dtype="U4"),
        groups=np.array(["patient001", "patient002", "patient003"], dtype="U10"),
    )
    X, y, groups = load_primary_data({
        "features_path": str(path), "feature_key": "wavelet_statistics", "classes": ["AMI", "HC"],
    })
    assert X.shape == (3, 4)
    assert y.tolist() == [0, 1, 0]
    assert groups.tolist() == ["patient001", "patient002", "patient003"]
