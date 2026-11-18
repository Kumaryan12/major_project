"""Prepare checksum-verified same-record before/during STAFF III pairs."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import re
from threading import local
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import scipy
from scipy.signal import resample_poly
import sklearn
from sklearn.model_selection import train_test_split
import wfdb
import yaml

from . import acs_ecg_bank, external, overlap_screen, signal, supervised_source_overlap
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash

RELEASES = (
    'https://physionet.org/files/staffiii/1.0.0/',
    'https://archive.physionet.org/physiobank/database/staffiii/',
)
INVENTORY_SHA256 = 'e4c71aa1b47b8fd17b1549b5f650770df63f4cb795388d129c6ca78cd7a78269'
WORKBOOK = 'STAFF-III-Database-Annotations.xlsx'
VESSELS = ('LAD', 'LCX', 'RCA')
THREAD = local()


def read(path):
    return json.loads(path.read_text())


def fetch(root, name, digest):
    path = root / name
    if path.is_file():
        if file_hash(path) != digest:
            raise ValueError(f'Existing published input fails checksum: {name}')
        return path
    if not hasattr(THREAD, 'session'):
        THREAD.session = requests.Session()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    for base in RELEASES * 2:
        try:
            with THREAD.session.get(base + name, timeout=(8, 30), stream=True) as response:
                response.raise_for_status()
                with temporary.open('wb') as out:
                    for chunk in response.iter_content(256 * 1024):
                        out.write(chunk)
            if file_hash(temporary) != digest:
                raise ValueError(f'Published input failed checksum: {name}')
            temporary.replace(path)
            return path
        except requests.RequestException as error:
            last_error = error
    raise RuntimeError(f'Public download failed: {name}') from last_error


def load_inventory(root):
    inventory = fetch(root, 'SHA256SUMS.txt', INVENTORY_SHA256)
    entries = [line.split(maxsplit=1) for line in inventory.read_text().splitlines()]
    return {name: digest for digest, name in entries}


def vessel_rows(workbook):
    """Read the fixed published sheet's BI slots and vessel/timing columns."""
    with ZipFile(workbook) as archive:
        strings = [''.join(t.text or '' for t in item.findall('.//{*}t'))
                   for item in ElementTree.fromstring(archive.read('xl/sharedStrings.xml'))]
        sheet = ElementTree.fromstring(archive.read('xl/worksheets/sheet1.xml'))
    slots = (('G', 'H', 'I'), ('K', 'L', 'M'), ('O', 'P', 'Q'), ('S', 'T', 'U'), ('V', 'W', 'X'))
    result = {}
    for row in sheet.findall('.//{*}row'):
        if int(row.get('r')) < 11:
            continue
        cells = {}
        for cell in row.findall('{*}c'):
            text = cell.findtext('{*}v')
            if text is not None:
                col = re.match(r'[A-Z]+', cell.attrib['r']).group()
                cells[col] = strings[int(text)] if cell.get('t') == 's' else text
        for ordinal, (file_col, label_col, duration_col) in enumerate(slots, 1):
            declared = cells.get(file_col, '').strip().casefold()
            if not declared:
                continue
            match = re.fullmatch(r'(\d+)([a-z])', declared)
            if match is None or int(cells['A']) != int(match[1]):
                raise ValueError('Unexpected STAFF III workbook recording ID')
            record = f'data/{int(match[1]):03d}{match[2]}'
            label = cells.get(label_col, '').upper()
            groups = set(re.findall(r'\b(?:LAD|LCX|RCA|CIRC|LM)\b', label))
            groups = {'LCX' if g == 'CIRC' else g for g in groups}
            vessel = next(iter(groups)) if len(groups) == 1 else ''
            raw_start = cells.get(duration_col, '').split(';')[0].strip()
            start = float(raw_start) if re.fullmatch(r'\d+(?:\.\d+)?', raw_start) else None
            result.setdefault(record, []).append((ordinal, vessel, start))
    return result


def paired_event(annotation, length, windows):
    events = [(int(p), label.strip().casefold()) for p, label in zip(annotation.sample, annotation.aux_note)]
    first = next((i for i, (_, name) in enumerate(events) if name == 'balloon inflation'), None)
    if first is None:
        return None, 'no_inflation'
    start = events[first][0]
    bounded = next((p for p, name in events[first+1:] if name in ('balloon deflation', 'balloon inflation')), None)
    before, during = [[start + int(1000 * x) for x in interval] for interval in windows]
    if before[0] < 0:
        return None, 'short_baseline'
    if bounded is None or during[1] > min(bounded, length):
        return None, 'unbounded_inflation'
    if any('contrast' in name and before[0] <= p < during[1] for p, name in events):
        return None, 'annotated_contrast'
    return (start, before, during), 'timing_candidate'


def metadata(config):
    root = Path(config['raw_dir'])
    inventory = load_inventory(root)
    workbook = fetch(root, WORKBOOK, inventory[WORKBOOK])
    labels = vessel_rows(workbook)
    records = sorted(path[:-6] for path in inventory if path.endswith('.event'))
    if len(records) != 142:
        raise ValueError('Published annotation inventory changed')

    def inspect(record):
        paths = [fetch(root, record + ext, inventory[record + ext]) for ext in ('.hea', '.event')]
        stem = str(paths[0].with_suffix(''))
        header = wfdb.rdheader(stem)
        ann = wfdb.rdann(stem, 'event')
        if header.fs != 1000 or not set(external.ECG_LEADS).issubset({n.lower() for n in header.sig_name}):
            raise ValueError('Unexpected STAFF III lead layout or sampling rate')
        result, status = paired_event(ann, header.sig_len, config['windows_seconds'])
        if result is None:
            return status, None
        patient = int(Path(record).name[:3])
        start, before, during = result
        if patient in config['excluded_source_patients']:
            return 'publisher_lead_warning', None
        choices = labels.get(record, [])
        if not choices:
            return 'no_workbook_vessel', None
        ordinal, vessel, start_sec = choices[0]
        if vessel not in VESSELS or start_sec is None or abs(start / 1000 - start_sec) > 1.1:
            return 'vessel_or_annotation_mismatch', None
        return 'labeled_timing_pair', dict(record=record, patient=patient, vessel=vessel, ordinal=ordinal,
                                           before=before, during=during, source_sha256=inventory[record + '.dat'])

    with ThreadPoolExecutor(max_workers=config['download_workers']) as pool:
        result = list(pool.map(inspect, records))
    status = dict(Counter(value for value, _ in result))
    selected = [row for _, row in result if row is not None]
    initial = len(selected) + status.get('publisher_lead_warning', 0) + status.get('no_workbook_vessel', 0) + status.get('vessel_or_annotation_mismatch', 0)
    if initial != 65:
        raise ValueError('The recorded timing-only audit no longer reproduces')
    counts = {v: len({r['patient'] for r in selected if r['vessel'] == v}) for v in VESSELS}
    print('Verified annotations:', status, 'eligible patient support:', counts, flush=True)
    return selected, status


def channel_indices(header, stem):
    names = [n.lower() for n in header.sig_name]
    if (len(names) != 9 or len(set(names)) != 9 or not set(external.ECG_LEADS).issubset(names)
            or header.fs != 1000 or set(header.file_name) != {stem + '.dat'}
            or set(header.fmt) != {'16'} or set(header.byte_offset) != {512}
            or set(header.samps_per_frame) != {1} or not set(header.skew).issubset({0, None})
            or not all(str(u).lower() == 'mv' for u in header.units)
            or not np.isfinite(header.adc_gain).all() or min(header.adc_gain) <= 0
            or not np.isfinite(header.baseline).all()):
        raise ValueError('Unexpected STAFF III physical encoding')
    return [names.index(lead) for lead in external.ECG_LEADS]


def physical_samples(raw, header, columns):
    x = np.asarray(raw[..., columns], dtype=np.float64)
    return (x - np.asarray(header.baseline)[columns]) / np.asarray(header.adc_gain)[columns]


def fetch_range(origin, name, lower, upper, size):
    if not hasattr(THREAD, 'session'):
        THREAD.session = requests.Session()
    expected = f'bytes {lower}-{upper}/{size}'
    for _ in range(3):
        try:
            with THREAD.session.get(origin + name, headers={'Range': f'bytes={lower}-{upper}'},
                                    timeout=(8, 25), stream=True) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                    raise ValueError('Unexpected range response from public waveform release')
                response.raw.decode_content = True
                payload = response.raw.read(upper-lower+2)
            if len(payload) != upper-lower+1:
                raise ValueError('Incomplete waveform byte range')
            return payload
        except (requests.RequestException, OSError) as error:
            last_error = error
    raise RuntimeError(f'Unable to retrieve public waveform range: {name}') from last_error


def waveform_windows(root, item):
    path = root / item['record']
    header = wfdb.rdheader(str(path))
    columns = channel_indices(header, path.name)
    waveform = path.with_suffix('.dat')
    cache = path.with_suffix('.paired.npz')
    receipt = path.with_suffix('.paired.json')
    context = dict(source_sha256=item['source_sha256'], header_sha256=file_hash(path.with_suffix('.hea')),
                   before=item['before'], during=item['during'])
    if receipt.is_file():
        state = read(receipt)
        if state['context'] != context or state['cache_sha256'] != file_hash(cache):
            raise ValueError('STAFF III clip scope or content changed')
        with np.load(cache, allow_pickle=False) as saved:
            raw = saved['samples']
        method = state['method']
    else:
        digital = []
        method = 'publisher_file_hash' if waveform.is_file() else 'identical_version_and_archive_ranges'
        if waveform.is_file() and file_hash(waveform) != item['source_sha256']:
            raise ValueError('Complete recording failed publisher checksum')
        for a,b in (item['before'],item['during']):
            lower, upper = 512 + 18*a, 512 + 18*b - 1
            if waveform.is_file():
                with waveform.open('rb') as handle:
                    handle.seek(lower)
                    payload = handle.read(upper-lower+1)
            else:
                name = item['record'] + '.dat'
                received = [fetch_range(origin,name,lower,upper,512+18*header.sig_len) for origin in RELEASES]
                if received[0] != received[1]:
                    raise ValueError('Waveform endpoint range copies disagree')
                payload = received[0]
            digital.append(np.frombuffer(payload,dtype='<i2').reshape(-1,9))
        raw = np.stack(digital)
        if waveform.is_file():
            for ix,(a,b) in enumerate((item['before'],item['during'])):
                expected = wfdb.rdrecord(str(path),sampfrom=a,sampto=b,channels=columns).p_signal
                if not np.array_equal(physical_samples(raw[ix],header,columns),expected):
                    raise ValueError('Range decoder does not match complete WFDB decoding')
        temp = cache.with_suffix('.partial.npz')
        np.savez_compressed(temp,samples=raw)
        temp.replace(cache)
        _save_json(receipt,dict(context=context,cache_sha256=file_hash(cache),method=method))
    if raw.shape != (2,10000,9) or raw.dtype != np.dtype('<i2'):
        raise ValueError('Unexpected digital waveform window shape')
    if np.any(raw == -32768):
        return None, method, file_hash(receipt)
    return physical_samples(raw,header,columns), method, file_hash(receipt)


def download_waveforms(config, selected):
    root = Path(config['raw_dir'])
    with ThreadPoolExecutor(max_workers=config['download_workers']) as pool:
        jobs = pool.map(lambda item: waveform_windows(root, item), selected)
        for index, _ in enumerate(jobs, 1):
            if index % 10 == 0 or index == len(selected):
                print(f'Verified source waveform pairs {index}/{len(selected)}', flush=True)


def crops(ecg, signal_config, config):
    xyz = external.kors_transform(ecg, list(external.ECG_LEADS))
    if not np.isfinite(xyz).all() or np.any(np.std(xyz, axis=0) < 1e-6):
        return None
    clean = signal.denoise_vcg(xyz, signal_config['denoise_wavelet'], signal_config['denoise_level'])
    peaks = signal.detect_r_peaks(clean, 1000, signal_config['rpeak_refractory_ms'],
                                 signal_config['rpeak_integration_ms'], signal_config['rpeak_threshold_scale'])
    complete = peaks[(peaks >= 250) & (peaks + 400 < len(clean))]
    if len(complete) < config['minimum_complete_beats']:
        return None
    chosen = complete[np.linspace(0, len(complete)-1, config['clips_per_window'], dtype=int)]
    waves = np.stack([clean[p-250:p+401].T for p in chosen]).astype(np.float32)
    return waves if np.isfinite(waves).all() else None


def decode_pairs(config, selected, settings):
    root = Path(config['raw_dir'])
    examples, skipped, methods, audits = [], Counter(), Counter(), {}
    for row in sorted(selected, key=lambda r: (r['patient'], r['ordinal'], r['record'])):
        windows, method, digest = waveform_windows(root, row)
        methods[method] += 1
        audits[row['record']] = digest
        if windows is None or not np.isfinite(windows).all():
            skipped['invalid_ecg_window'] += 1
            continue
        beats = [crops(w, settings, config) for w in windows]
        if any(b is None for b in beats):
            skipped['insufficient_xyz_quality_or_beats'] += 1
            continue
        examples.append((row, np.stack(beats), windows))
    first_by_patient = {}
    for item in examples:
        first_by_patient.setdefault(item[0]['patient'], item)
    skipped['additional_patient_pairs'] = len(examples) - len(first_by_patient)
    return list(first_by_patient.values()), dict(skipped), dict(methods), _hash(audits)


def prepare(config, mode):
    selected, statuses = metadata(config)
    if mode == 'download':
        download_waveforms(config, selected)
        return None
    target = yaml.safe_load(Path(config['target_config']).read_text())
    paper = Path(target['paper_config'])
    settings = yaml.safe_load(paper.read_text())['signal']
    root = Path(config['bank_dir'])
    root.mkdir(parents=True, exist_ok=True)
    download_waveforms(config, selected)
    pairs, qc, methods, waveform_receipt_hash = decode_pairs(config, selected, settings)
    if not pairs:
        raise ValueError('No paired source waveforms survived QC')
    rows = [r for r, _, _ in pairs]
    y = np.array([VESSELS.index(r['vessel']) for r in rows])
    groups = np.array([r['patient'] for r in rows])
    train, val = train_test_split(np.arange(len(y)), test_size=config['source_validation_fraction'],
                                  random_state=config['source_seed'], stratify=y)
    train, val = np.sort(train), np.sort(val)
    supports = {phase: {v: int(np.count_nonzero(y[split] == i)) for i,v in enumerate(VESSELS)}
                for phase, split in [('train', train), ('validation', val)]}
    scope = dict(config=config, inventory_sha256=INVENTORY_SHA256, input_rows_sha256=_hash(selected),
                 retained_rows_sha256=_hash(rows), paper_sha256=file_hash(paper),
                 hashes={Path(m).name: file_hash(m) for m in (__file__, acs_ecg_bank.__file__, external.__file__,
                          signal.__file__, overlap_screen.__file__, supervised_source_overlap.__file__)},
                 numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, wfdb=wfdb.__version__,
                 waveform_receipts_sha256=waveform_receipt_hash)
    screened_rows = pd.DataFrame({'data_sha256': [r['source_sha256'] for r in rows for _ in range(2)]})
    continuous = [resample_poly(w, 1, 4, axis=0).astype(np.float32) for _, _, windows in pairs for w in windows]
    if len({r['source_sha256'] for r in rows}) != len(rows) or len(set(map(array_hash, continuous))) != len(continuous):
        raise ValueError('Retained source patients have duplicated signals')
    screen = yaml.safe_load(Path(config['copy_screen_config']).read_text())
    scope['screen_configuration_sha256'] = file_hash(config['copy_screen_config'])
    result = acs_ecg_bank.screen_copies(screen, target, screened_rows, continuous, scope, root)
    support_ok = (min(supports['train'].values()) >= config['minimum_training_per_vessel']
                  and min(supports['validation'].values()) >= config['minimum_validation_per_vessel'])
    arrays = dict(waves=np.stack([b for _,b,_ in pairs]), y=y, groups=groups, train=train, validation=val)
    receipt = dict(scope=scope, timing_status=statuses, qc=qc, patient_support=supports, waveform_methods=methods,
                   recording_copy_screen=result, source_support_passed=support_ok,
                   array_sha256={k: array_hash(v) for k,v in arrays.items()})
    receipt_path = root / 'receipt.json'
    if mode == 'verify':
        if read(receipt_path) != receipt:
            raise ValueError('Independent STAFF III bank replay differs')
        load(config, verified=False)
        _save_json(root / 'verified.json', dict(receipt_sha256=file_hash(receipt_path)))
    elif receipt_path.is_file():
        if read(receipt_path) != receipt:
            raise ValueError('Pinned staff bank changed')
    else:
        temp = root / 'bank.partial.npz'
        np.savez(temp, **arrays)
        temp.replace(root / 'bank.npz')
        _save_json(receipt_path, receipt)
    print('Source QC:', qc, 'support:', supports, 'screen:', result['passed'], flush=True)
    return receipt


def load(config, verified=True):
    root = Path(config['bank_dir'])
    receipt = read(root / 'receipt.json')
    if receipt['scope']['config'] != config:
        raise ValueError('Bank config changed')
    if verified and read(root / 'verified.json')['receipt_sha256'] != file_hash(root / 'receipt.json'):
        raise ValueError('Verified STAFF III bank receipt changed')
    with np.load(root / 'bank.npz', allow_pickle=False) as bank:
        arrays = {k: bank[k] for k in bank.files}
    if {k: array_hash(v) for k,v in arrays.items()} != receipt['array_sha256']:
        raise ValueError('Frozen STAFF III arrays changed')
    return arrays, receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('download', 'prepare', 'verify'))
    parser.add_argument('--config', default='configs/staffiii_pair_transfer_v1.yaml')
    args = parser.parse_args()
    prepare(yaml.safe_load(Path(args.config).read_text()), args.mode)
