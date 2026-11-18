"""Fit and independently restore matched mean and change source encoders."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from . import benchmark, neural, staffiii_pair_bank as bank
from . import supervised_source_train as tools
from .contrastive import _progress, initial_state_hash
from .multiview_bank import array_hash
from .signal_audit import file_hash

ARMS = ('mean', 'change')


class PairEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        base = neural.VCGResNet(len(bank.VESSELS), config['widths'], config['pool_bins'], config['dropout'])
        self.encoder, self.classifier = base.encoder, base.classifier

    def forward(self, waves, arm):
        # Encode the same beats in both arms, then average beats within each time window.
        features = self.encoder(waves.flatten(0, 2)).reshape(*waves.shape[:3], -1).mean(2)
        before, during = features.unbind(1)
        if arm == 'change':
            pooled = during - before
        elif arm == 'mean':
            pooled = (during + before) * 0.5
        else:
            raise ValueError('Invalid source pooling arm')
        return self.classifier(pooled)


def initial_model(target, seed):
    neural._seed(seed, target)
    return PairEncoder(target)


def evaluate(model, arm, waves, val, device):
    model.eval()
    with torch.no_grad():
        scores = model(waves[val].to(device), arm).softmax(-1).cpu().numpy()
    if not np.isfinite(scores).all() or not np.allclose(scores.sum(1), 1):
        raise ValueError('Invalid source validation probabilities')
    return scores


def fit(config, target, root, waves, labels, val, schedule, scope, arm, verify):
    folder = root / 'review' / 'source' / arm
    folder.mkdir(parents=True, exist_ok=True)
    path, checkpoint, probability_path = (folder / n for n in ('audit.json', 'model.pt', 'probabilities.npz'))
    model = initial_model(target, config['source_seed'])
    context = dict(arm=arm, fingerprint=neural._hash(scope), initial_sha256=initial_state_hash(model),
                   schedule_sha256=array_hash(schedule), validation_rows_sha256=array_hash(val))
    device = neural._device(target)
    model.to(device)
    saved = path.exists()
    if saved or verify:
        evidence = bank.read(path)
        if (evidence['context'] != context or evidence['model_sha256'] != file_hash(checkpoint)
                or evidence['probability_sha256'] != file_hash(probability_path)):
            raise ValueError('Source model or probability evidence changed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['context'] != context:
            raise ValueError('Source checkpoint scope changed')
        model.load_state_dict(state['state'], strict=True)
        history = evidence['history']
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=target['learning_rate'], weight_decay=target['weight_decay'])
        history = []
        for epoch, updates in enumerate(schedule, 1):
            model.train()
            losses = []
            for rows in updates:
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(waves[rows].to(device), arm), labels[rows].to(device))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite paired source loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            history.append(dict(epoch=epoch, sampled_pairs_sha256=array_hash(updates), loss=float(np.mean(losses))))
            _progress(root, state='source_training', arm=arm, epoch=epoch, total_epochs=len(schedule))
    if (len(history) != len(schedule) or any(not torch.isfinite(v).all() for v in model.state_dict().values())
            or any(r['epoch'] != i or r['sampled_pairs_sha256'] != array_hash(u) or not np.isfinite(r['loss'])
                   for i, (r,u) in enumerate(zip(history,schedule),1))):
        raise ValueError('Invalid source model state or exposure receipt')
    scores = evaluate(model, arm, waves, val, device)
    if saved:
        with np.load(probability_path, allow_pickle=False) as frozen:
            if not np.array_equal(frozen['rows'], val) or not np.allclose(frozen['p'], scores, atol=2e-5, rtol=2e-5):
                raise ValueError('Source validation probabilities changed')
            scores = frozen['p']
    else:
        tools.save_checkpoint(checkpoint, dict(context=context, state=model.state_dict()))
        temp = probability_path.with_suffix('.partial.npz')
        np.savez_compressed(temp, rows=val, p=scores)
        temp.replace(probability_path)
        neural._save_json(path, dict(context=context, history=history,
                                     model_sha256=file_hash(checkpoint), probability_sha256=file_hash(probability_path)))
    if verify:
        export = folder / 'encoder.pt'
        state = {k: v.detach().cpu() for k,v in model.encoder.state_dict().items()}
        if export.is_file():
            old = torch.load(export, map_location='cpu', weights_only=True)
            if old['context'] != context or old['state'].keys() != state.keys() or any(not torch.equal(v, old['state'][k]) for k,v in state.items()):
                raise ValueError('Replayed source export differs')
        else:
            tools.save_checkpoint(export, dict(context=context, state=state))
        f1 = f1_score(labels[val].numpy(), scores.argmax(1), labels=np.arange(len(bank.VESSELS)), average='macro', zero_division=0)
        return dict(encoder_weights_sha256=initial_state_hash(model.encoder), encoder_file_sha256=file_hash(export),
                    audit_sha256=file_hash(path), validation_vessel_macro_f1=float(f1))
    return None


def run(config, smoke, verify):
    arrays, evidence = bank.load(config)
    if not evidence['source_support_passed'] or not evidence['recording_copy_screen']['passed']:
        raise ValueError('STAFF III bank did not pass the declared support and recording screen')
    target = yaml.safe_load(Path(config['target_config']).read_text())
    root = Path(config['output_dir']) / ('smoke' if smoke else 'complete')
    root.mkdir(parents=True, exist_ok=True)
    train, val = arrays['train'], arrays['validation']
    x = arrays['waves']
    norm = tools.fit_normalizer(x.reshape(len(x), -1, 3, 651), np.full(len(x), x.shape[1]*x.shape[2]), arrays['groups'], train)
    mean, scale = (np.asarray(norm[k], np.float32)[:, None] for k in ('mean', 'scale'))
    waves = torch.from_numpy((x-mean)/scale)
    labels = torch.from_numpy(arrays['y'])
    weights = benchmark.patient_balanced_weights(arrays['y'][train], arrays['groups'][train])
    rng = np.random.default_rng(config['source_seed'])
    updates = (1, 2) if smoke else (config['source_epochs'],config['updates_per_epoch'])
    schedule = rng.choice(train, size=(*updates,config['pairs_per_batch']), p=weights/weights.sum())
    scope = dict(config=config, smoke=smoke, bank_sha256=neural._hash(evidence), norm=norm,
                 training_rows_sha256=array_hash(train), code_sha256=file_hash(__file__),
                 tool_code_sha256=file_hash(tools.__file__), neural_code_sha256=file_hash(neural.__file__),
                 target_config_sha256=file_hash(config['target_config']), device=str(neural._device(target)),
                 torch_version=str(torch.__version__))
    guard = root / 'source_manifest.json'
    if (guard.exists() or verify) and bank.read(guard) != scope:
        raise ValueError('Source optimization scope changed')
    if not guard.exists():
        neural._save_json(guard, scope)
    runs = {arm: fit(config,target,root,waves,labels,val,schedule,scope,arm,verify) for arm in ARMS}
    receipts = [bank.read(root/'review'/'source'/arm/'audit.json')['context'] for arm in ARMS]
    for key in ('initial_sha256', 'schedule_sha256', 'validation_rows_sha256'):
        if receipts[0][key] != receipts[1][key]:
            raise ValueError('Source arms do not match')
    if verify:
        result = dict(complete=True, models=2, smoke=smoke, source_scope_sha256=file_hash(guard),
                      bank_sha256=neural._hash(evidence), fits=runs, source_patient_support=evidence['patient_support'],
                      recording_copy_screen=evidence['recording_copy_screen'])
        summary = root / 'source_summary.json'
        if summary.exists() and bank.read(summary) != result:
            raise ValueError('Source summary did not reproduce')
        neural._save_json(summary, result)
        print('Verified source vessel macro F1:', {a: v['validation_vessel_macro_f1'] for a,v in runs.items()}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/staffiii_pair_transfer_v1.yaml')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--verify',action='store_true')
    args = p.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()),args.smoke,args.verify)
