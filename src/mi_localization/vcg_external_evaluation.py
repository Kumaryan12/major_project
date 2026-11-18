"""Score a locked partner cohort with one verified measured-XYZ pilot ensemble."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from . import benchmark, preprocessing_study
from .neural import _device, _save_json, _seed, VCGResNet
from .residual_context_report import patient_weights, weighted_f1
from .signal_audit import file_hash
from .supervised_transfer import restore_norm
from .vcg_expansion_bank import crop, decode
from .vcg_external_intake import CLASSES, read_cohort, validate


def read(path):
    return json.loads(path.read_text())


def checked_members(study_root, arm):
    scope = read(study_root / 'study_manifest.json')
    summary = read(study_root / 'summary.json')
    verified = read(study_root / 'verification_manifest.json')
    config = scope['config']
    if (scope['smoke'] or summary.get('complete') is not True
            or summary['study_fingerprint'] != scope['fingerprint'] or verified['fingerprint'] != scope['fingerprint']
            or summary['verification_manifest_sha256'] != file_hash(study_root / 'verification_manifest.json')
            or summary['report_sha256'] != file_hash(study_root / 'REPORT.md')
            or tuple(config['classes']) != CLASSES or arm not in summary['ensemble_metrics']
            or len(config['training_seeds']) != 3 or config['outer_splits'] != 10
            or any(file_hash(Path(path)) != digest for path, digest in scope['code'].items())):
        raise ValueError('Require a complete verified pilot arm with the original six classes')
    members = []
    for fold in range(1, config['outer_splits'] + 1):
        for seed in config['training_seeds']:
            name = f'fold{fold:02d}_seed{seed}_{arm}'
            directory = study_root / 'review' / name
            audit_path = directory / 'audit.json'
            audit_hash = file_hash(audit_path)
            audit = read(audit_path)
            model = directory / 'model.pt'
            model_hash = file_hash(model)
            context = audit['context']
            if (audit_hash != verified['fit_audit_hashes'][name] or model_hash != audit['files']['model.pt']
                    or context['fingerprint'] != scope['fingerprint'] or context['arm'] != arm
                    or context['fold'] != fold or context['declared_seed'] != seed):
                raise ValueError('Verified model or membership changed')
            members.append((model, audit, model_hash))
    return scope, members


def evaluate(manifest, signals_root, ptb_root, study_root, arm, output, device='auto'):
    scope, members = checked_members(study_root, arm)
    paper_path = Path(scope['config']['paper_config'])
    if file_hash(paper_path) != scope['bank']['paper_sha256']:
        raise ValueError('Original waveform preprocessing configuration changed')
    output.mkdir(parents=True, exist_ok=True)
    cohort = validate(manifest, signals_root, ptb_root, output / 'intake.json')
    if not cohort['external_has_all_classes']:
        raise ValueError('The locked external six-class evaluation requires coverage of all six classes')
    config = dict(scope['config'], device=device)
    lock = dict(arm=arm, study_fingerprint=scope['fingerprint'],
                checkpoint_hashes=[digest for _, _, digest in members], cohort=cohort,
                evaluation_code_sha256=file_hash(__file__), device=str(_device(config)))
    lock_path = output / 'evaluation_scope.json'
    if lock_path.exists():
        if read(lock_path) != lock:
            raise ValueError('Arm, cohort or evaluation scope changed. Use a new evaluation namespace.')
    else:
        _save_json(lock_path, lock)
    rows, _ = read_cohort(manifest, set())
    rows = [row for row in rows if row['partition'] == 'external_test' and row['review_status'] == 'confirmed']
    paper = yaml.safe_load(paper_path.read_text())
    blocks, groups, labels = [], [], []
    for row in rows:
        beats = crop(decode(signals_root / row['record']), paper)
        if not len(beats) or not np.isfinite(beats).all():
            raise ValueError('An eligible external recording has no valid finite crops')
        blocks.append(beats)
        groups.extend([row['patient_id']] * len(beats))
        labels.extend([CLASSES.index(row['label'])] * len(beats))
    waves = np.concatenate(blocks)
    group_array, truth = np.asarray(groups), np.asarray(labels)
    ensemble = np.zeros((len(waves), len(CLASSES)), dtype=np.float64)
    _seed(config['seed'], config)
    torch_device = _device(config)
    for path, audit, digest in members:
        if file_hash(path) != digest:
            raise ValueError('A frozen checkpoint changed during evaluation')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['context'] != audit['context']:
            raise ValueError('Checkpoint content does not match its audit')
        model = VCGResNet(len(CLASSES), tuple(config['widths']), config['pool_bins'], config['dropout'])
        model.load_state_dict(state['model'], strict=True)
        model.to(torch_device)
        fit = audit['fit']
        norm = restore_norm(fit['normalization'], fit['normalizer_fit_beats'])
        ensemble += preprocessing_study.predict_waves(
            model, norm, waves, np.zeros(len(waves), dtype=int), np.arange(len(waves)), config) / len(members)
    patient_ids, patient_truth, probabilities = benchmark.aggregate_patients(ensemble, truth, group_array)
    weights = patient_weights(patient_truth, np.arange(len(CLASSES)), config['bootstrap_draws'], config['seed'])
    distribution = weighted_f1(patient_truth, probabilities.argmax(axis=1), np.arange(len(CLASSES)), weights)
    result = dict(arm=arm, cohort_scope_sha256=file_hash(lock_path), patients=len(patient_ids),
                  recordings=len(rows), beats=len(waves), equally_weighted_models=len(members),
                  patient_metrics=benchmark.metric_bundle(patient_truth, probabilities, list(CLASSES)),
                  patient_macro_f1_interval=np.quantile(distribution, [.025, .975]).tolist(),
                  interval_method='Class-stratified patient bootstrap of fixed external predictions')
    receipt = output / 'summary.json'
    if receipt.exists():
        if read(receipt) != result:
            raise ValueError('Saved external evaluation does not reproduce')
    else:
        _save_json(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('signals', type=Path)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ptb-root', type=Path, default=Path('data/raw/ptbdb'))
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    result = evaluate(args.manifest, args.signals, args.ptb_root, args.study, args.arm, args.output, args.device)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
