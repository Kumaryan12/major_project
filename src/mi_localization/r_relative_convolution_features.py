"""Pool frozen short convolution responses within fixed R-relative windows."""
import numpy as np
import torch
from torch.nn import functional as F

from .rocket_features import aggregate, check_waves

ARMS = ('whole_beat', 'shuffled', 'aligned')


def signed_rms(response):
    """Positive and negative components of thresholded response energy."""
    return np.stack([
        np.sqrt(np.square(np.maximum(response, 0)).mean(axis=-1)),
        np.sqrt(np.square(np.minimum(response, 0)).mean(axis=-1)),
    ], axis=-1)


def permute_windows(values, permutations):
    if (values.shape[:2] != permutations.shape or
            not np.all(np.sort(permutations, axis=1) == np.arange(values.shape[1]))):
        raise ValueError('Each record needs a complete window permutation')
    return values[np.arange(len(values))[:, None], permutations]


def validate_windows(config, waveform_length):
    windows = np.asarray(config['windows_ms'])
    if (windows.shape != (4, 2) or not np.issubdtype(windows.dtype, np.integer)
            or not np.all(windows[:-1, 1] == windows[1:, 0])
            or len(set(windows[:, 1] - windows[:, 0])) != 1
            or not np.all(windows[:, 1] > windows[:, 0])):
        raise ValueError('Require four adjacent, positive, equal-duration windows')
    if (config['dilations'] != [1, 4] or not 0 < config['kernels_per_window'] <= 21
            or not 0 <= config['r_index'] < waveform_length):
        raise ValueError('Invalid short response and kernel contract')
    start, stop = windows[0, 0] + config['r_index'], windows[-1, 1] + config['r_index']
    padding = 4 * max(config['dilations'])
    if start < padding or stop > waveform_length - 1 - padding:
        raise ValueError('R-relative windows must fit on valid raw and differenced response centers')


def convolution_blocks(templates, transform, permutations, config):
    """Use the exact-partition frozen raw/difference kernels and thresholds."""
    templates = check_waves(templates)
    validate_windows(config, templates.shape[-1])
    x = torch.from_numpy(templates / transform.scale)
    rng = np.random.default_rng(config['kernel_selection_seed'])
    windows = np.asarray(config['windows_ms']) + config['r_index']
    n_slots, anchor = len(windows), config['kernels_per_window']
    collected = {arm: [] for arm in ARMS}
    kernel_ids = []
    group = 0
    for view, dilations in enumerate(transform.multirocket.dilations):
        wave = x if view == 0 else torch.diff(x, axis=-1)
        if not set(config['dilations']).issubset(dilations):
            raise ValueError('Source transform is missing a declared short dilation')
        for dilation in dilations:
            wi = group
            group += 1
            if dilation not in config['dilations']:
                continue
            weights, biases = transform.multirocket.weights[wi], transform.multirocket.biases[wi]
            chosen = rng.choice(len(weights), n_slots * anchor, replace=False)
            kernel_ids.append(chosen.tolist())
            # Valid convolution only. Position i corresponds to original center i+4*dilation.
            center_offset = 4 * dilation
            chunks, global_chunks = [], []
            with torch.inference_mode():
                for block in wave.split(transform.config['chunk_size']):
                    response = F.conv1d(block, torch.from_numpy(weights[chosen]), dilation=dilation).numpy()
                    response = response - biases[chosen][None, :, None]
                    whole_beat = signed_rms(response).reshape(len(block), n_slots, anchor, 2)
                    aligned = np.stack([
                        signed_rms(response[:, :anchor, left-center_offset:right-center_offset])
                        for left, right in windows
                    ], axis=1)
                    chunks.append(aligned)
                    global_chunks.append(whole_beat)
            aligned = np.concatenate(chunks)
            whole_beat = np.concatenate(global_chunks)
            if aligned.shape != whole_beat.shape:
                raise ValueError('Augmented tensor widths differ')
            collected['aligned'].append(aligned.reshape(len(x), -1))
            collected['shuffled'].append(permute_windows(aligned, permutations).reshape(len(x), -1))
            collected['whole_beat'].append(whole_beat.reshape(len(x), -1))
    if len(kernel_ids) != 2 * len(config['dilations']):
        raise ValueError('Raw and differenced response groups must both be present')
    features = {name: np.concatenate(parts, axis=1) for name, parts in collected.items()}
    if any(not np.isfinite(v).all() for v in features.values()):
        raise ValueError('Nonfinite R-relative convolution summaries')
    return features, kernel_ids


def patient_blocks(bank, transform, patient_rows, seed, config):
    indices = bank.records_for(bank.ids[patient_rows])
    rng = np.random.default_rng(config['window_shuffle_seed'] + seed)
    permutations = np.stack([rng.permutation(len(config['windows_ms'])) for _ in bank.record_patients])
    values, kernel_ids = convolution_blocks(bank.templates[indices], transform, permutations[indices], config)
    return (aggregate(values, bank.record_patients[indices], bank.ids[patient_rows]),
            indices, kernel_ids, permutations[indices])
