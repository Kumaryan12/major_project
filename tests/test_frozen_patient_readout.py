import subprocess
import sys


def test_streamed_feature_means_match_frozen_encoder_and_linear_head():
    # Keep native CPU Torch initialization independent of the mixed numerical test process.
    code = """
import numpy as np
import torch
from scipy.special import softmax
from mi_localization import later_recording_distillation as source
from mi_localization.frozen_patient_readout import encode_patient
from mi_localization.contrastive import initial_state_hash

model = source.new_model(dict(device='cpu', torch_threads=2, classes=list('abcdef'),
                              widths=[16, 32, 64], pool_bins=4, dropout=.2), 42)
model.eval()
model.requires_grad_(False)
waves = np.random.default_rng(7).normal(size=(5,3,651)).astype(np.float32)
rows = np.array([0,2,4])
normalization = {'wave_mean_':[.1,-.1,0.], 'wave_scale_':[.5,2.,1.]}
before = initial_state_hash(model)
features, probabilities, pooled = encode_patient(model,waves,rows,normalization,2)
x = torch.from_numpy((waves[rows]-np.array(normalization['wave_mean_'],np.float32)[:,None])
                     /np.array(normalization['wave_scale_'],np.float32)[:,None])
with torch.inference_mode():
    bins = model.encoder[:8](x)
    h = model.encoder[8:](bins)
    logits = model.classifier(h)
np.testing.assert_allclose(features[256],bins.numpy().mean(0),atol=2e-6,rtol=2e-6)
np.testing.assert_allclose(features[64],h.numpy().mean(0),atol=2e-6,rtol=2e-6)
np.testing.assert_allclose(probabilities,logits.softmax(1).numpy().mean(0),atol=2e-6,rtol=2e-6)
np.testing.assert_allclose(pooled,softmax(logits.numpy().mean(0)),atol=2e-6,rtol=2e-6)
waves[1] = 1000
unchanged = encode_patient(model,waves,rows,normalization,2)
np.testing.assert_array_equal(probabilities,unchanged[1])
assert initial_state_hash(model) == before
assert all(parameter.grad is None for parameter in model.parameters())
try:
    encode_patient(model,waves,rows,normalization,2,np.zeros((len(rows),6)))
except ValueError as error:
    assert 'per-beat' in str(error)
else:
    raise AssertionError('A corrupted source prediction must fail')
"""
    subprocess.run([sys.executable, '-c', code], check=True)
