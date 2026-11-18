from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from mi_localization.acs_artery_pretraining import source_schedule
from mi_localization.acs_ecg_bank import ARTERIES, labels_for_train


def test_auxiliary_mask_keeps_parent_group_and_masks_ambiguous_records():
    labels = [dict(Patient_id=p, OMI=omi, **{seg: 0 for group in ARTERIES.values() for seg in group})
              for p, omi in [('a', 1), ('b', 1), ('c', 1), ('d', 0), ('e', 1), ('e', 1), ('f', 1)]]
    labels[0].update(PLAD=1, DB=1, PRCA=2)
    labels[1].update(MLCX=1, OM=1)
    labels[2].update(PLAD=1, PRCA=1)
    labels[3].update(PRCA=1)
    labels[4].update(PLAD=1)
    labels[5].update(PRCA=1)
    labels[6].update(LM=1)
    target, conflict = labels_for_train(pd.DataFrame(labels))
    assert target.tolist() == ['LAD', 'LCX', '', '', '', '', '']
    assert conflict.tolist() == [False, False, False, False, True, True, False]


def test_matched_source_schedule_is_training_only_and_reproducible():
    arrays = dict(omi=np.array([0, 0, 0, 1, 1]), groups=np.array(['a', 'a', 'b', 'c', 'heldout']))
    rows = np.array([0, 1, 2, 3])
    config = dict(source_seed=42, source_epochs=8, source_samples_per_epoch=4096, source_batch_size=32)
    sampled = source_schedule(config, arrays, rows)
    np.testing.assert_array_equal(sampled, source_schedule(config, arrays, rows))
    assert sampled.shape == (8, 128, 32)
    assert np.isin(sampled, rows).all()
    frequency = [np.mean(arrays['groups'][sampled] == key) for key in ('a', 'b', 'c')]
    np.testing.assert_allclose(frequency, [.25, .25, .5], atol=.02)


def test_masked_auxiliary_torch_objective_in_fresh_process():
    root = Path(__file__).resolve().parents[1]
    code = '''
import torch
from mi_localization.acs_artery_pretraining import objectives
from mi_localization.neural import _seed
_seed(42, {'device': 'cpu', 'torch_threads': 2})
logits = torch.zeros((3, 4), requires_grad=True)
omi = torch.tensor([1, 1, 0])
artery = torch.tensor([2, -1, -1])
binary, auxiliary = objectives(logits, omi, artery)
(binary + .5 * auxiliary).backward()
assert not logits.grad[:1, 1:].eq(0).any()
assert logits.grad[1:, 1:].eq(0).all()
assert not logits.grad[:, 0].eq(0).any()
assert objectives(logits, omi, torch.full_like(artery, -1))[1].item() == 0
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
