"""Audit additional measured Frank recordings without changing target eligibility."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import resample_poly
import wfdb
import yaml

from . import data, overlap_screen, paired_bank, signal
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .overlap_screen import rank_candidates
from .signal_audit import file_hash

LEADS = ['vx', 'vy', 'vz']


def decode(path):
    """Read only the three measured Frank channels in the prescribed orientation."""
    head = wfdb.rdheader(str(path))
    names = [v.lower() for v in head.sig_name]
    if any(names.count(lead) != 1 for lead in LEADS):
        raise ValueError('Expected exactly one each of VX, VY and VZ')
    record = wfdb.rdrecord(str(path), channels=[names.index(lead) for lead in LEADS])
    values = record.p_signal
    if (record.fs != 1000 or [v.lower() for v in record.units] != ['mv'] * 3
            or values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all()):
        raise ValueError('Expected finite measured XYZ in mV at 1000 Hz')
    return values


def crop(values, paper):
    settings = paper['signal']
    cleaned, _ = paired_bank.denoise_ecg(values, settings['denoise_wavelet'], settings['denoise_level'])
    peaks = signal.detect_r_peaks(
        cleaned, 1000, settings['rpeak_refractory_ms'],
        settings['rpeak_integration_ms'], settings['rpeak_threshold_scale'])
    return signal.extract_beats(cleaned, peaks, paper['data']['before_r'], paper['data']['after_r'])[0]


def retrieval_templates(beats, points):
    if not len(beats):
        return np.zeros((4, 3 * points), dtype=np.float32), False
    sample = np.stack([np.median(beats, axis=0), beats[0], beats[len(beats) // 2], beats[-1]])
    x = np.linspace(0, beats.shape[-1] - 1, points)
    compact = np.array([[np.interp(x, np.arange(len(lead)), lead) for lead in row] for row in sample])
    compact -= compact.mean(axis=-1, keepdims=True)
    compact /= np.maximum(np.linalg.norm(compact, axis=-1, keepdims=True), 1e-12)
    compact = compact.reshape(4, -1)
    compact /= np.maximum(np.linalg.norm(compact, axis=1, keepdims=True), 1e-12)
    return compact.astype(np.float32), True


class CopySearch:
    """Screen a fixed middle excerpt for a shifted, scaled copy in a recording."""

    def __init__(self, values, samples):
        values = np.asarray(values, dtype=np.float64)
        self.length = len(values)
        self.samples = samples
        if self.length < samples:
            raise ValueError('Recording is too short for the copy screen')
        self.nfft = next_fast_len(len(values) + samples - 1)
        self.fft = rfft(values, self.nfft, axis=0)
        sums = np.vstack([np.zeros((1, 3)), np.cumsum(values, axis=0)])
        squares = np.vstack([np.zeros((1, 3)), np.cumsum(values**2, axis=0)])
        vsum = sums[samples:] - sums[:-samples]
        self.variance = np.maximum(squares[samples:] - squares[:-samples] - vsum**2 / samples, 0)

    def matches(self, values, config):
        width = self.samples
        if len(values) < width:
            return False
        start = (len(values) - width) // 2
        query = values[start:start + width].astype(np.float64, copy=True)
        query -= query.mean(axis=0)
        energy = np.square(query).sum(axis=0)
        if np.any(np.sqrt(energy / width) < config['minimum_lead_sd_mv']):
            return False
        response = irfft(self.fft * rfft(query, self.nfft, axis=0).conj(), self.nfft, axis=0)
        denom = np.sqrt(self.variance * energy)
        rows = self.length - width + 1
        r = np.divide(response[:rows], denom, out=np.zeros_like(denom), where=denom > 1e-18)
        r = np.clip(r, -1, 1)
        return bool(np.any((r.mean(axis=1) >= config['copy_mean_r'])
                           & (r.min(axis=1) >= config['copy_min_r'])))


def source_file(record, original, dest, url, allow_download):
    old = original / record
    new = dest / record
    new.parent.mkdir(parents=True, exist_ok=True)
    header = new.with_suffix('.hea')
    if not header.exists():
        shutil.copyfile(old.with_suffix('.hea'), header)
    if file_hash(header) != file_hash(old.with_suffix('.hea')):
        raise ValueError('Source header changed')
    raw = new.with_suffix('.xyz')
    if not raw.is_file() or raw.stat().st_size == 0:
        cached = old.with_suffix('.xyz')
        if cached.is_file() and cached.stat().st_size:
            temporary = raw.with_suffix('.xyz.part')
            shutil.copyfile(cached, temporary)
            temporary.replace(raw)
        elif allow_download:
            data.download_file(f'{url}/{record}.xyz', raw)
        else:
            raise FileNotFoundError('Native XYZ files are missing. Run audit with --download.')
    return new


def load(config):
    root = Path(config['source_bank_dir'])
    manifest = json.loads((root / 'manifest.json').read_text())
    files = manifest['bank_files']
    if any(file_hash(root / name) != digest for name, digest in files.items()):
        raise ValueError('Native source bank content changed')
    waves = np.load(root / 'waveforms.npy', mmap_mode='r', allow_pickle=False)
    groups = np.load(root / 'groups.npy', allow_pickle=False)
    if (waves.shape != (len(groups), 3, 651) or not len(groups)
            or len(groups) != manifest['n_beats']
            or len(np.unique(groups)) != manifest['n_patients']
            or not np.isfinite(waves).all()):
        raise ValueError('Invalid audited native source bank')
    return waves, groups, manifest


def audit(config, download=False):
    paper = yaml.safe_load(Path(config['paper_config']).read_text())
    original = Path(paper['data']['raw_dir'])
    outfile = Path(config['source_audit_dir'])
    bank = Path(config['source_bank_dir'])
    target_folds = pd.read_csv(config['folds_path'])
    if target_folds.patient_id.duplicated().any():
        raise ValueError('Target patients must have one fixed fold')
    target = set(target_folds.patient_id.astype(str))
    beats = pd.read_csv(config['beats_path'])
    beats = beats[beats.label.isin(config['classes'])]
    if set(beats.patient_id.astype(str)) != target:
        raise ValueError('Frozen target metadata and folds differ')
    target_records = sorted(beats.record.unique())
    headers = sorted(original.glob('patient*/*.hea'))
    if len(headers) != 549 or len(target_records) != 391:
        raise ValueError('Expected the frozen 549 PTB headers and 391 target records')
    exclusions = set(paper['data'].get('excluded_patients', []))
    records = [str(h.relative_to(original).with_suffix('')) for h in headers
               if h.parent.name not in target | exclusions]
    if not records:
        raise ValueError('No separate native source patients found')
    dest = Path(config['source_raw_dir'])
    with ThreadPoolExecutor(max_workers=config['download_workers']) as pool:
        paths = list(pool.map(lambda record: source_file(
            record, original, dest, paper['data']['database_url'], download), records))
    print(f'Native source files ready: {len(paths)} recordings', flush=True)
    provenance = {
        'config': config, 'folds_sha256': file_hash(config['folds_path']),
        'beats_sha256': file_hash(config['beats_path']), 'paper_sha256': file_hash(config['paper_config']),
        'source_hashes_sha256': _hash({str(p): [file_hash(p.with_suffix('.hea')), file_hash(p.with_suffix('.xyz'))]
                                      for p in paths}),
        'target_raw_hashes_sha256': _hash({r: [file_hash((original / r).with_suffix('.hea')),
                                             file_hash((original / r).with_suffix('.xyz'))]
                                          for r in target_records}),
        'code': {str(Path(p)): file_hash(p) for p in (
            __file__, signal.__file__, overlap_screen.__file__, paired_bank.__file__, data.__file__)},
        'runtime': {'numpy': np.__version__, 'wfdb': wfdb.__version__},
    }
    fingerprint = _hash(provenance)
    if (bank / 'manifest.json').exists():
        _, _, saved = load(config)
        if saved['fingerprint'] != fingerprint:
            raise ValueError('Native audit provenance changed. Use a new namespace.')
        summary = json.loads((outfile / 'summary.json').read_text())
        if summary['fingerprint'] != fingerprint or summary['report_sha256'] != file_hash(outfile / 'REPORT.md'):
            raise ValueError('Native audit summary or report changed')
        return summary

    factor, remainder = divmod(1000, config['screen_rate'])
    if remainder or factor < 1:
        raise ValueError('Copy screen rate must divide 1000 Hz')
    width = config['screen_rate'] * config['screen_seconds']
    points = config['template_points']
    target_signals, feature_rows, availability, searches, raw_hashes = [], [], [], [], set()
    by_record = beats.groupby('record')
    for name in target_records:
        xyz = decode(original / name)
        raw_hashes.add(array_hash(xyz))
        cached_peaks = by_record.get_group(name).r_peak.to_numpy(dtype=int)
        template, available = retrieval_templates(signal.extract_beats(xyz, cached_peaks)[0], points)
        down = resample_poly(xyz, 1, factor, axis=0).astype(np.float32)
        target_signals.append(down)
        feature_rows.append(template)
        availability.append(available)
        searches.append(None if len(down) < width else CopySearch(down, width))
    target_features = np.stack(feature_rows)
    short = sum(search is None for search in searches)
    if short:
        raise ValueError('Some target recordings are shorter than the continuous-copy query')
    controls = 0
    for i in np.linspace(0, len(searches) - 1, 12, dtype=int):
        j = int(i)
        candidates = rank_candidates(target_features[j:j+1], target_features,
                                     [availability[j]], availability, config['shortlist_size'])[0]
        shifted = target_signals[j][7:-3] * np.array([.97, 1.05, 1.02]) + .02
        if j not in candidates or not searches[j].matches(shifted, config):
            raise ValueError('Known-copy retrieval/correlation control failed')
        controls += 1
    print(f'Native copy screen: {controls} controls passed', flush=True)
    candidates_by_patient = {}
    qc, quarantined, source_hashes = [], set(), {}
    comparisons = 0
    for index, (record, path) in enumerate(zip(records, paths), 1):
        patient = Path(record).parent.name
        try:
            xyz = decode(path)
        except ValueError as error:
            qc.append({'record': record, 'status': 'invalid_signal', 'reason': str(error)})
            continue
        down = resample_poly(xyz, 1, factor, axis=0).astype(np.float32)
        status = 'eligible'
        if len(down) < width or np.any(np.std(down, axis=0) < config['minimum_lead_sd_mv']):
            status = 'short_or_low_lead_variation'
        raw_peaks = signal.detect_r_peaks(xyz, 1000, 250, 150, .06) if len(down) >= width else np.empty(0, dtype=int)
        template, available = retrieval_templates(signal.extract_beats(xyz, raw_peaks)[0], points)
        shortlist = rank_candidates(template[None], target_features, [available],
                                    availability, config['shortlist_size'])[0]
        signature = array_hash(xyz)
        exact = signature in raw_hashes
        owner = source_hashes.setdefault(signature, patient)
        if owner != patient:
            quarantined.update((owner, patient))
        comparable = len(down) >= width and np.all(
            np.std(down[(len(down) - width)//2:(len(down) - width)//2 + width], axis=0)
            >= config['minimum_lead_sd_mv'])
        near = False
        if comparable:
            comparisons += len(shortlist)
            near = any([searches[i].matches(down, config) for i in shortlist])
        if exact or near:
            status = 'possible_target_copy'
            quarantined.add(patient)
        elif not comparable and status == 'eligible':
            status = 'insufficient_continuous_copy_coverage'
        elif status == 'eligible':
            windows = crop(xyz, paper)
            if len(windows) < config['minimum_beats'] or not np.isfinite(windows).all():
                status = 'insufficient_valid_crops'
            else:
                candidates_by_patient.setdefault(patient, []).append(windows)
        qc.append({'record': record, 'status': status, 'exact_target_copy': exact, 'near_target_copy': near})
        if index % 40 == 0:
            print(f'Native audit {index}/{len(records)} recordings', flush=True)

    patients = sorted(set(candidates_by_patient) - quarantined)
    if len(patients) < 2 or set(patients) & target:
        raise ValueError('Native bank needs screened and disjoint source patients')
    blocks = [np.concatenate(candidates_by_patient[p]) for p in patients]
    grouped = np.repeat(np.asarray(patients), [len(x) for x in blocks])
    bank.mkdir(parents=True, exist_ok=True)
    for name, values in (('waveforms', np.concatenate(blocks)), ('groups', grouped)):
        temp = bank / f'{name}.partial.npy'
        np.save(temp, values, allow_pickle=False)
        temp.replace(bank / f'{name}.npy')
    files = {name: file_hash(bank / name) for name in ('waveforms.npy', 'groups.npy')}
    summary = dict(complete=True, fingerprint=fingerprint, source_candidate_records=len(records),
                   source_candidate_patients=len(set(Path(r).parent.name for r in records)),
                   n_patients=len(patients), n_records=sum(len(candidates_by_patient[p]) for p in patients),
                   n_beats=len(grouped), target_records_screened=len(target_records),
                   continuous_comparisons=comparisons, known_copy_controls_passed=controls,
                   patients_quarantined=len(quarantined),
                   qc_counts=pd.Series([row['status'] for row in qc]).value_counts().to_dict())
    (outfile / 'review').mkdir(parents=True, exist_ok=True)
    _save_json(outfile / 'review' / 'record_qc.json', qc)
    report = outfile / 'REPORT.md'
    report.write_text(
        '# Native measured VCG source audit\n\n'
        f"Screened {len(records)} source records against {len(target_records)} target records. "
        f"Kept {summary['n_patients']} source patients, {summary['n_records']} source recordings and "
        f"{summary['n_beats']:,} beats. All {controls} shifted/gain-adjusted controls passed. "
        f"Quarantined {len(quarantined)} source patient IDs with target copy flags or shared exact source recordings.\n\n"
        'The public PTB headers define patient-directory membership. The screen checks exact decoded XYZ '
        'hashes against every target recording and applies a continuous three-lead correlation check to '
        'a beat-template shortlist. Clinical identity and disease labels require different evidence. '
        'Training source records use signed mV channels, basic variation and crop checks, and source-only '
        'R detection. Original target patient labels, beats and R positions follow the frozen configuration.\n',
        encoding='utf-8')
    _save_json(outfile / 'summary.json', dict(summary, report_sha256=file_hash(report)))
    _save_json(bank / 'manifest.json', dict(summary, provenance=provenance, bank_files=files))
    return dict(summary, report_sha256=file_hash(report))
