"""Validate a private, adjudicated measured-Frank cohort before model evaluation."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

import numpy as np
import wfdb

from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash
from .vcg_expansion_bank import decode

CLASSES = ('AMI', 'ALMI', 'ASMI', 'IMI', 'ILMI', 'HC')
ASSESSMENT = ('cardiac_mri', 'clinical_adjudication', 'control_review')
FIELDS = ('record', 'patient_id', 'site', 'partition', 'label', 'review_status', 'evidence_type')
PARTITIONS = ('development', 'external_test')


def read_cohort(manifest, ptb_patients):
    with manifest.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or []) != FIELDS:
            raise ValueError(f'Require manifest columns {FIELDS}')
        rows = list(reader)
    if not rows:
        raise ValueError('No partner recordings provided')
    identifiers = re.compile(r'[a-zA-Z0-9_-]+')
    assignments, records = {}, set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError('Every manifest row must match the declared columns')
        patient, site, partition = row['patient_id'], row['site'], row['partition']
        if not identifiers.fullmatch(patient) or not identifiers.fullmatch(site):
            raise ValueError('Require pseudonymous patient and site identifiers')
        if patient in ptb_patients or partition not in PARTITIONS:
            raise ValueError('Invalid partition or patient ID shared with PTB')
        record = Path(row['record'])
        if (record.is_absolute() or not record.parts or record.suffix or '..' in record.parts
                or any(not identifiers.fullmatch(part) for part in record.parts) or record in records):
            raise ValueError('Record paths must be unique relative WFDB stems')
        records.add(record)
        status, label, evidence = row['review_status'], row['label'], row['evidence_type']
        if status == 'confirmed':
            if label not in CLASSES or evidence not in ASSESSMENT:
                raise ValueError('Confirmed labels need a declared class and review evidence')
            if (label == 'HC') != (evidence == 'control_review'):
                raise ValueError('Control and infarct review evidence must match the declared label')
        elif status not in ('indeterminate', 'outside_endpoint') or label != '' or evidence != '':
            raise ValueError('Keep unconfirmed records in intake with an empty label and evidence field')
        item = (site, partition, status, label)
        if assignments.setdefault(patient, item) != item:
            raise ValueError('Patient has inconsistent site, split, review status or label')
    return rows, assignments


def exact_ptb_hashes(original):
    headers = sorted(original.glob('patient*/*.hea'))
    if len(headers) != 549:
        raise ValueError('Require the original PTB header inventory for the identity screen')
    patients = {path.parent.name for path in headers}
    signals = set()
    for header in headers:
        if header.with_suffix('.xyz').is_file():
            signals.add(array_hash(decode(header.with_suffix(''))))
    return patients, signals


def validate(manifest, signals_root, ptb_root, output):
    ptb_patients, ptb_signals = exact_ptb_hashes(ptb_root)
    rows, assignments = read_cohort(manifest, ptb_patients)
    root = signals_root.resolve()
    wave_hashes, encoded_hashes = {}, {}
    for row in rows:
        path = (root / row['record']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('A recording resolves outside the private signal root')
        if not path.with_suffix('.hea').resolve().is_relative_to(root):
            raise ValueError('A WFDB header resolves outside the private signal root')
        header = wfdb.rdheader(str(path))
        files = getattr(header, 'file_name', None)
        if not files or any(not (path.parent / name).resolve().is_relative_to(root) for name in files):
            raise ValueError('Expected a single-segment WFDB recording with local signal files')
        values = decode(path)
        digest = array_hash(values)
        key = row['patient_id']
        if digest in ptb_signals or wave_hashes.setdefault(digest, key) != key:
            raise ValueError('Potential duplicate across PTB or partner patient IDs')
        if len(values) < 651 or np.any(values.std(axis=0) < 0.001):
            raise ValueError('A partner recording failed basic duration or XYZ variation checks')
        encoded_hashes[row['record']] = [file_hash(path.with_suffix('.hea')), digest]
    counts = {part: Counter(label for _, assigned, status, label in assignments.values()
                            if assigned == part and status == 'confirmed')
              for part in PARTITIONS}
    counts = {part: {label: n[label] for label in CLASSES} for part, n in counts.items()}
    summary = dict(
        scope='private intake and candidate eligibility',
        manifest_sha256=file_hash(manifest),
        reviewed_signals_sha256=_hash(encoded_hashes),
        patients=len(assignments), recordings=len(rows),
        patients_by_review_status=dict(Counter(item[2] for item in assignments.values())),
        confirmed_patients_by_partition_and_class=counts,
        external_has_all_classes=all(counts['external_test'].values()),
        cached_ptb_xyz_unique_signals_screened=len(ptb_signals),
    )
    if output.exists() and json.loads(output.read_text()) != summary:
        raise ValueError('Intake receipt changed. Use a new version of the receipt.')
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        _save_json(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('signals', type=Path)
    parser.add_argument('--ptb-root', type=Path, default=Path('data/raw/ptbdb'))
    parser.add_argument('--output', type=Path,
                        default=Path('data/processed/partner_vcg_intake_v1/summary.json'))
    args = parser.parse_args()
    print(json.dumps(validate(args.manifest, args.signals, args.ptb_root, args.output), indent=2))


if __name__ == '__main__':
    main()
