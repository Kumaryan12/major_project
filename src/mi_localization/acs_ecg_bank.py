"""Prepare and independently replay a bounded angiography-linked ACS-ECG bank."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split
import wfdb
import yaml

from . import external, overlap_screen, signal, supervised_source_overlap as expanded
from . import supervised_source_bank as old_bank
from .neural import _hash, _save_json
from .signal_audit import file_hash
from .supervised_source_audit import safe_path

ARTERIES = {
    'LM': ['LM'], 'LAD': ['PLAD', 'MLAD', 'DLAD', 'DB'],
    'LCX': ['PLCX', 'MLCX', 'DLCX', 'OM'], 'RCA': ['PRCA', 'MRCA', 'DRCA'],
}
AUXILIARY = ('LAD', 'LCX', 'RCA')
ASSETS = {
    'metadata_archive': (296769, '1cb46279c0e68e6512bd39214fc56528'),
    'raw_archive': (1277577332, 'acea6ca86a2d0b937ecfe7d1df6d30a2'),
}
TARGET_LEADS = {'i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'}


def read(path):
    return json.loads(path.read_text())


def verified_archive(path, key):
    expected_size, expected_md5 = ASSETS[key]
    if path.stat().st_size != expected_size:
        raise ValueError(f'Incorrect archive size for {key}')
    with path.open('rb') as handle:
        if hashlib.file_digest(handle, 'md5').hexdigest() != expected_md5:
            raise ValueError(f'Incorrect archive checksum for {key}')
    return file_hash(path)


def labels_for_train(table):
    segment_codes = [segment for segments in ARTERIES.values() for segment in segments]
    if (not table['OMI'].isin([0, 1]).all()
            or not table[segment_codes].isin([0, 1, 2]).all().all()):
        raise ValueError('Unexpected published OMI/artery annotation values')
    flags = pd.DataFrame({a: table[c].eq(1).any(axis=1) for a, c in ARTERIES.items()})
    exact = flags.sum(axis=1).eq(1) & table.OMI.eq(1)
    single = flags.idxmax(axis=1).where(exact, '')
    contradictory_patient = single.loc[exact].groupby(table['Patient_id']).nunique().gt(1)
    conflict = table.Patient_id.isin(contradictory_patient.index[contradictory_patient])
    return single.where(~conflict & single.isin(AUXILIARY), ''), conflict


def cohort(config, smoke):
    archive = Path(config['metadata_archive'])
    metadata_hash = verified_archive(archive, 'metadata_archive')
    with ZipFile(archive) as z:
        train = pd.read_csv(z.open('CSV/train.csv'), dtype={'Patient_id': 'string', 'ecg_row_record': 'string'})
        holdout = pd.read_csv(z.open('CSV/test.csv'), usecols=['Patient_id', 'ecg_row_record'], dtype='string')
    train, holdout = train.rename(columns={'ecg_row_record': 'waveform'}), holdout.rename(columns={'ecg_row_record': 'waveform'})
    for table in (train, holdout):
        if table.Patient_id.isna().any() or not table.waveform.str.fullmatch(r'\d{5}\.dat').fillna(False).all() or not table.waveform.is_unique:
            raise ValueError('Unsafe or duplicate metadata waveform key')
    if (len(train), train.Patient_id.nunique(), len(holdout), holdout.Patient_id.nunique()) != (17960, 17018, 1995, 1891):
        raise ValueError('Published metadata counts changed')
    if (set(train.Patient_id) & set(holdout.Patient_id) or set(train.waveform) & set(holdout.waveform)):
        raise ValueError('Publisher test IDs overlap labeled IDs')
    train['artery'], conflict = labels_for_train(train)
    patient_binary = train.groupby('Patient_id').OMI.max()
    patient_artery = train.loc[train.artery.ne('')].groupby('Patient_id').artery.first()
    strata = patient_binary.map({0: 'negative', 1: 'other_omi'})
    strata.update(patient_artery)
    positive = strata.index[strata.ne('negative')].to_numpy(dtype=str)
    negative = strata.index[strata.eq('negative')].to_numpy(dtype=str)
    if len(positive) != 1139 or int(train.loc[conflict, 'Patient_id'].nunique()) != 2:
        raise ValueError('Metadata preflight no longer reproduces')
    rng = np.random.default_rng(config['source_seed'])
    selected = np.sort(np.r_[positive, rng.choice(negative, size=config['negative_patients'], replace=False)])
    tr, va = train_test_split(selected, test_size=config['validation_fraction'], random_state=config['source_seed'],
                              stratify=strata.loc[selected])
    if smoke:
        tr = np.concatenate([np.sort(tr[strata.loc[tr].to_numpy() == s])[:2] for s in sorted(strata.unique())])
        va = np.concatenate([np.sort(va[strata.loc[va].to_numpy() == s])[:2] for s in sorted(strata.unique())])
    chosen = train.loc[train.Patient_id.isin(np.r_[tr, va]), ['Patient_id', 'waveform', 'OMI', 'artery']].copy()
    chosen['role'] = np.where(chosen.Patient_id.isin(tr), 'train', 'validation')
    chosen = chosen.sort_values('waveform').reset_index(drop=True)
    return chosen, metadata_hash


def decode(path):
    header = wfdb.rdheader(str(path))
    names = [n.lower() for n in header.sig_name]
    if (header.fs != 500 or header.sig_len != 5000 or len(names) != 12 or set(names) != TARGET_LEADS
            or set(header.file_name) != {path.name + '.dat'}
            or set(header.fmt) != {'16'} or set(u.lower() for u in header.units) != {'mv'}
            or not np.isfinite(header.adc_gain).all() or min(header.adc_gain) <= 0
            or path.with_suffix('.dat').stat().st_size != 5000 * 12 * 2):
        raise ValueError('Unexpected ACS-ECG header, units, sample count, or encoding')
    record = wfdb.rdrecord(str(path))
    if record.p_signal.shape != (5000, 12):
        raise ValueError('Decoded ECG shape differs from the declared WFDB signal')
    return record.p_signal[:, [names.index(x) for x in external.ECG_LEADS]]


def crops(ecg, settings, limit):
    if not np.isfinite(ecg).all():
        return np.empty((0, 3, 651), dtype=np.float32), 'nonfinite'
    xyz = resample_poly(external.kors_transform(ecg, list(external.ECG_LEADS)), 2, 1, axis=0)
    if np.any(np.std(xyz, axis=0) < 1e-6):
        return np.empty((0, 3, 651), dtype=np.float32), 'flat_xyz'
    clean = signal.denoise_vcg(xyz, settings['denoise_wavelet'], settings['denoise_level'])
    peaks = signal.detect_r_peaks(clean, 1000, settings['rpeak_refractory_ms'],
                                 settings['rpeak_integration_ms'], settings['rpeak_threshold_scale'])
    complete = peaks[(peaks >= 250) & (peaks + 400 < len(clean))]
    if not len(complete):
        return np.empty((0, 3, 651), dtype=np.float32), 'no_complete_beat'
    position = np.linspace(0, len(complete) - 1, min(len(complete), limit), dtype=int)
    return np.stack([clean[p-250:p+401].T for p in complete[position]]).astype(np.float32), 'accepted'


def source_rows(config, selected, settings, screen):
    waves, ecgs, rows = [], [], []
    with ZipFile(config['raw_archive']) as archive, tempfile.TemporaryDirectory() as directory:
        files = archive.namelist()
        if len(files) != len(set(files)):
            raise ValueError('Duplicate ZIP members')
        files = set(files)
        for index, row in enumerate(selected.itertuples(), 1):
            datum = Path(row.waveform)
            header = datum.with_suffix('.hea')
            path = Path(directory) / datum.stem
            for name in (header, datum):
                member = f'row_data/{name}'
                if member not in files:
                    raise ValueError('Metadata record is missing from the waveform archive')
                (Path(directory) / name).write_bytes(archive.read(member))
            ecg = decode(path)
            beats, reason = crops(ecg, settings, config['beats_per_record'])
            encoded_hash = file_hash(path.with_suffix('.dat'))
            keep = reason == 'accepted' and np.isfinite(beats).all()
            if reason == 'accepted' and not keep:
                raise ValueError('Nonfinite processed ACS-ECG beats')
            if keep:
                waves.append(beats)
                ecgs.append(resample_poly(ecg, 1, 500 // screen['screen_rate'], axis=0).astype(np.float32))
            rows.append(dict(ecg_id=int(datum.stem), patient_id=str(row.Patient_id), role=row.role,
                             omi=int(row.OMI), artery=row.artery, accepted=bool(keep), reason=reason,
                             data_sha256=encoded_hash, hea_sha256=file_hash(path.with_suffix('.hea')),
                             physical_xyz_sha256=old_bank.array_hash(ecg.astype('<f4')) if keep else '',
                             waveform_index=len(waves)-1 if keep else -1))
            (Path(directory) / header).unlink()
            (Path(directory) / datum).unlink()
            if index % 200 == 0 or index == len(selected):
                print(f'Decoded source records {index}/{len(selected)}', flush=True)
    ledger, duplicates = old_bank.quarantine(pd.DataFrame(rows))
    retained = ledger.loc[ledger.accepted].reset_index(drop=True)
    dropped = Counter(ledger.loc[~ledger.accepted, 'reason'])
    return retained, waves, ecgs, dict(dropped=dict(dropped), duplicate_groups=len(duplicates),
                                       ledger_sha256=_hash(ledger.to_dict('records')))


def screen_copies(config, target_config, retained, ecgs, scope, base):
    target_paper = yaml.safe_load(Path(target_config['paper_config']).read_text())
    beats = pd.read_csv(target_config['beats_path'], usecols=['record', 'label', 'patient_id'])
    beats = beats.loc[beats.label.isin(target_config['classes'])]
    records = sorted(beats.record.unique())
    if (len(records), beats.patient_id.nunique(), len(beats)) != (391, 164, 55863):
        raise ValueError('Target raw recording scope changed')
    target_dir = target_paper['data']['raw_dir']
    items = [dict(id=r, path=str(safe_path(target_dir, r)), sampling_rate=1000) for r in records]
    inventory = expanded.inventory(items)
    hashes = {row['data_sha256'] for row in inventory}
    encoded = int(retained.data_sha256.isin(hashes).sum())
    cache_dir = base / 'screen_targets'
    cache_dir.mkdir(parents=True, exist_ok=True)
    screen = dict(config, cache_dir=str(cache_dir))
    fingerprint = _hash(dict(scope=scope, target_inventory=inventory))
    targets = [overlap_screen._cache_record(item, screen, fingerprint) for item in inventory]
    paths, features, available = zip(*targets)
    features, available = np.stack(features), np.asarray(available)
    count, controls = overlap_screen._controls(paths, features, available, screen)
    if count is None:
        raise ValueError('The known-copy controls failed on the target recording bank')
    sources = [overlap_screen.templates(ecg, screen) for ecg in ecgs]
    sf, sa = zip(*sources)
    candidates = expanded.retrieve(np.stack(sf), features, np.asarray(sa), available, count, screen)
    queries = [[] for _ in paths]
    for source, target_indices in enumerate(candidates):
        for target in target_indices:
            queries[target].append(source)

    def compare(item):
        path, group = item
        if not group:
            return []
        with np.load(path, allow_pickle=False) as saved:
            target = overlap_screen.CorrelationTarget(saved['ecg'], config['screen_rate'] * config['query_seconds'])
        return [target.compare(ecgs[index], config) for index in group]

    with ThreadPoolExecutor(max_workers=screen['workers']) as pool:
        values = [row for batch in pool.map(compare, zip(paths, queries)) for row in batch]
    found = sum(row['candidate'] for row in values)
    insufficient = sum(row['active_leads'] < config['minimum_active_leads'] for row in values)
    return dict(checked_source_records=len(ecgs), checked_target_records=len(items), shortlist_size=count,
                known_copy_controls=len(controls), continuous_comparisons=len(values),
                candidates=int(found), encoded_matches=encoded, insufficient_lead_pairs=int(insufficient),
                passed=not (found or encoded or insufficient), patient_identity_cleared=False,
                target_inventory_sha256=_hash(inventory), comparisons_sha256=_hash(values))


def prepare(config, smoke=False, verify=False):
    selected, metadata_hash = cohort(config, smoke)
    raw_hash = verified_archive(Path(config['raw_archive']), 'raw_archive')
    target = yaml.safe_load(Path(config['target_config']).read_text())
    paper_file = Path(target['paper_config'])
    settings = yaml.safe_load(paper_file.read_text())['signal']
    original_screen = yaml.safe_load(Path(config['copy_screen_config']).read_text())
    screen_keys = ('seed', 'screen_rate', 'query_seconds', 'template_points', 'candidate_counts',
                   'candidate_batch_size', 'positive_controls', 'mean_correlation_threshold',
                   'minimum_lead_correlation', 'minimum_active_leads', 'minimum_lead_sd_mv', 'workers')
    screen = {key: original_screen[key] for key in screen_keys}
    overlap_screen.validate_config(screen)
    base = Path(config['bank_dir']) / ('smoke' if smoke else 'full')
    base.mkdir(parents=True, exist_ok=True)
    scope = dict(config=config, smoke=smoke, metadata_sha256=metadata_hash, waveform_archive_sha256=raw_hash,
                 paper_sha256=file_hash(paper_file), selected_sha256=_hash(selected.to_dict('records')),
                 code={Path(p).name: file_hash(p) for p in (
                     __file__, external.__file__, signal.__file__, expanded.__file__,
                     overlap_screen.__file__, old_bank.__file__)},
                 numpy=np.__version__, pandas=pd.__version__, wfdb=wfdb.__version__)
    saved_path = base / 'receipt.json'
    if (saved_path.is_file() or verify) and read(saved_path)['scope'] != scope:
        raise ValueError('ACS-ECG bank scope changed')
    if saved_path.is_file() and not verify:
        return
    retained, values, ecgs, details = source_rows(config, selected, settings, screen)
    indices = retained.waveform_index.to_numpy()
    split = {role: retained.loc[retained.role.eq(role)] for role in ('train', 'validation')}
    if set(split['train'].patient_id) & set(split['validation'].patient_id):
        raise ValueError('ACS-ECG patient split overlaps after QC')
    limit = 1 if smoke else config['minimum_artery_training_patients']
    if (min(split['train'].groupby('artery').patient_id.nunique().reindex(AUXILIARY, fill_value=0)) < limit
            or any(set(frame.omi) != {0, 1} or not set(AUXILIARY).issubset(frame.artery) for frame in split.values())):
        raise ValueError('Insufficient retained source label support')
    for role, frame in split.items():
        details[role] = dict(records=len(frame), patients=int(frame.patient_id.nunique()),
                             omi_positive_records=int(frame.omi.sum()),
                             artery_patients={a: int(frame.loc[frame.artery.eq(a), 'patient_id'].nunique()) for a in AUXILIARY})
    if not smoke:
        print('Screening retained ACS-ECG recordings for waveform copies against the PTB bank', flush=True)
        details['recording_copy_screen'] = screen_copies(screen, target, retained, [ecgs[i] for i in indices], scope, base)
    print(f"Source bank retained {len(retained)}/{len(selected)} records", flush=True)
    counts = np.array([len(values[i]) for i in indices], dtype=np.int64)
    waves = np.zeros((len(counts), config['beats_per_record'], 3, 651), dtype=np.float32)
    for offset, (index, count) in enumerate(zip(indices, counts)):
        waves[offset, :count] = values[index]
    arrays = dict(waves=waves, counts=counts, groups=retained.patient_id.to_numpy(dtype=str),
                  roles=retained.role.to_numpy(dtype=str), omi=retained.omi.to_numpy(dtype=np.int64),
                  artery=np.array([AUXILIARY.index(a) if a else -1 for a in retained.artery], dtype=np.int64))
    receipt = dict(scope=scope, summary=details, array_sha256={k: old_bank.array_hash(v) for k, v in arrays.items()})
    if verify:
        if read(saved_path) != receipt:
            raise ValueError('Independent source decoding or screening differs')
        load(config, smoke, require_verified=False)
        _save_json(base / 'verified.json', dict(receipt_sha256=file_hash(saved_path)))
        if not smoke and not details['recording_copy_screen']['passed']:
            raise ValueError('Recording copy screen requires review before training')
    else:
        path = base / 'bank.partial.npz'
        np.savez(path, **arrays)
        path.replace(base / 'bank.npz')
        _save_json(saved_path, receipt)


def load(config, smoke, require_verified=True):
    base = Path(config['bank_dir']) / ('smoke' if smoke else 'full')
    receipt = read(base / 'receipt.json')
    if require_verified:
        if read(base / 'verified.json')['receipt_sha256'] != file_hash(base / 'receipt.json'):
            raise ValueError('Bank requires independently verified source decoding')
        if not smoke and receipt['summary']['recording_copy_screen']['passed'] is not True:
            raise ValueError('Training requires a passing recording copy screen')
    with np.load(base / 'bank.npz', allow_pickle=False) as bundle:
        arrays = {k: bundle[k] for k in bundle.files}
    if {k: old_bank.array_hash(v) for k, v in arrays.items()} != receipt['array_sha256']:
        raise ValueError('Source bank arrays changed')
    return arrays, receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/acs_artery_transfer_v1.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    prepare(yaml.safe_load(Path(args.config).read_text()), args.smoke, args.verify)
