import json

import numpy as np

from mi_localization.earliest_record_convolution import waveform_reference
from mi_localization.multiview_bank import array_hash
from mi_localization.signal_audit import file_hash


def test_reference_uses_verified_review_directory_and_exact_earliest_rows(tmp_path):
    name = 'fold01_seed42_supervised'
    folder = tmp_path / 'review' / name
    folder.mkdir(parents=True)
    chosen_rows = np.array([0, 2, 3])
    fit_rows, query_rows = chosen_rows[:1], chosen_rows[1:]
    probabilities = np.array([[.1, .4, .2, .1, .1, .1], [.1, .5, .1, .1, .1, .1]], dtype=np.float32)
    snapshot = folder / 'probabilities.npz'
    np.savez(snapshot, rows=query_rows, probabilities=probabilities)
    audit = folder / 'audit.json'
    audit.write_text(json.dumps({
        'context': {'fit_rows_sha256': array_hash(fit_rows),
                    'query_rows_sha256': array_hash(query_rows)},
        'probabilities_sha256': file_hash(snapshot),
    }))
    data = {'earliest': chosen_rows, 'groups': np.array([0, 1, 1, 1]),
            'beat_labels': np.array([0, 1, 1, 1])}
    p = waveform_reference(tmp_path, data, 1, 42, np.array([0]), np.array([1]), {name: file_hash(audit)})
    np.testing.assert_array_equal(p[0], probabilities.mean(axis=0))
