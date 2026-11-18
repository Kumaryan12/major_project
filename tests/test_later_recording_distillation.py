import numpy as np
import pytest

from mi_localization.later_recording_distillation import shuffled_targets, teacher_partitions


def test_held_out_donors_share_teacher_partition_and_preserve_label():
    labels = np.repeat(np.arange(3), [4, 5, 8])
    patients = np.arange(len(labels))
    partitions = teacher_partitions(patients, labels, 3, 42)
    np.testing.assert_array_equal(np.sort(np.concatenate(partitions)), patients)
    for query in partitions:
        donors = query[shuffled_targets(query, labels, 731)]
        np.testing.assert_array_equal(np.sort(donors), query)
        np.testing.assert_array_equal(labels[donors], labels[query])
        assert np.all(donors != query)
        train = np.setdiff1d(patients, query)
        assert len(np.unique(labels[train])) == 3
        assert not np.intersect1d(donors, train).size


def test_split_rejects_class_too_small_for_both_matching_and_leakage_control():
    labels = np.repeat(np.arange(2), [3, 7])
    with pytest.raises(ValueError, match='held-out blocks'):
        teacher_partitions(np.arange(len(labels)), labels, 3, 42)
