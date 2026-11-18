"""Reconcile documented native-region mentions with the frozen measured-XYZ bank."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import yaml

from . import labels, vcg_expansion_bank
from .multiview_bank import array_hash
from .neural import _hash
from .signal_audit import file_hash
from .territory_study import DESCRIPTORS


def read(path):
    return json.loads(path.read_text())


def source_code(header):
    text = header.read_text(encoding='latin1')
    return labels.label_from_comments([line for line in text.splitlines() if line.startswith('#')])


def audit(config, base, target_groups):
    receipt = vcg_expansion_bank.audit(base)
    waves, groups, manifest = vcg_expansion_bank.load(base)
    if set(groups) & set(target_groups):
        raise ValueError('Native source and frozen target membership overlap')
    status_path = Path(base['source_audit_dir']) / 'review/record_qc.json'
    records = read(status_path)
    source_root = Path(base['source_raw_dir'])
    paper = yaml.safe_load(Path(base['paper_config']).read_text())
    by_patient = defaultdict(list)
    observed = set()
    for entry in records:
        record = entry['record']
        path = Path(record)
        patient = path.parent.name
        if record in observed:
            raise ValueError('Duplicate candidate record in native audit')
        observed.add(record)
        if entry['status'] == 'eligible' and patient in groups:
            header = (source_root / path).with_suffix('.hea')
            by_patient[patient].append((path, source_code(header), file_hash(header)))
    if set(by_patient) != set(groups) or sum(map(len, by_patient.values())) != manifest['n_records']:
        raise ValueError('The eligible record inventory does not match the frozen source bank')
    codings = config['native_source_regions']
    eligible = {}
    counts = Counter()
    supported_counts = Counter()
    skipped_complete_label = 0
    conflicting = 0
    candidates = 0
    matched_records = matched_beats = 0
    source_hashes = {}
    for patient, visits in sorted(by_patient.items()):
        codes = {value for _, value, _ in visits}
        if not codes.intersection(codings):
            continue
        candidates += 1
        if len(codes) != 1:
            conflicting += 1
            continue
        code = codes.pop()
        counts[code] += 1
        support = [int(region in codings[code]) for region in DESCRIPTORS]
        rows = np.flatnonzero(groups == patient)
        blocks = []
        for path, _, digest in sorted(visits):
            values = vcg_expansion_bank.decode(source_root / path)
            blocks.append(vcg_expansion_bank.crop(values, paper))
            source_hashes[str(path)] = digest
        if not np.array_equal(np.concatenate(blocks), waves[rows]):
            raise ValueError('Selected native waveform rows do not reproduce')
        matched_records += len(visits)
        matched_beats += len(rows)
        if not any(support):
            skipped_complete_label += 1
            continue
        supported_counts[code] += 1
        eligible[patient] = (rows, support)
    if (candidates != config['expected_source_patients_with_location'] or conflicting
            or len(eligible) != config['expected_supported_source_patients']):
        raise ValueError('Declared partial-label cohort differs from audited records')
    source_rows = np.concatenate([row for row, _ in eligible.values()])
    codes = np.asarray([tags for _, tags in eligible.values()], dtype=np.float32)
    sizes = [len(row) for row, _ in eligible.values()]
    report = dict(
        native_audit_fingerprint=receipt['fingerprint'],
        native_bank_manifest_sha256=file_hash(Path(base['source_bank_dir']) / 'manifest.json'),
        record_qc_sha256=file_hash(status_path), confirmed_header_sha256=_hash(source_hashes),
        original_source_codes=dict(counts), supported_source_codes=dict(supported_counts),
        patients_with_consistent_location=sum(counts.values()), patients_with_usable_mentions=len(eligible),
        patients_with_no_shared_region=skipped_complete_label, conflicting_candidate_patients=conflicting,
        reproduced_records=matched_records, reproduced_beats=matched_beats,
        supervised_source_beats=len(source_rows),
        region_patient_counts=dict(zip(DESCRIPTORS, codes.sum(axis=0).astype(int).tolist())),
        source_row_sha256=array_hash(source_rows), observed_tags_sha256=array_hash(codes),
    )
    pools = np.split(np.arange(len(source_rows)), np.cumsum(sizes)[:-1])
    return np.asarray(waves[source_rows]), pools, codes, report
