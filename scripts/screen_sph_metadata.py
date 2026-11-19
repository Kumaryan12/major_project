"""Read-only SPH release metadata screen; no signal downloads, mapping or fitting."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

EXPECTED = {'metadata.csv': 'c6f8dc5197758a3970aae554f8b6ee884d96656b121879bc0eb1adf367ea6d86',
            'code.csv': '4d22759fd19d37133c78d439a5726f9e0bc412cc17483c424926ce9abb33c8fc'}


def screen(folder):
    files = {}
    for name, expected in EXPECTED.items():
        path = folder / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError('Release metadata changed; use a new screen namespace')
        files[name] = dict(bytes=path.stat().st_size, sha256=digest)
    with (folder / 'metadata.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    with (folder / 'code.csv').open(encoding='utf-8-sig', newline='') as stream:
        dictionary = list(csv.DictReader(stream))
    terms = {row['Code']: row['Description'] for row in dictionary}
    regions = {row['Code']: row['Description'] for row in dictionary if row['Category'] == 'M'}
    patients, ids = set(), set()
    records, people, timing = {k: set() for k in regions}, {k: set() for k in regions}, {k: Counter() for k in regions}
    any_records, any_patients, multiregion = set(), set(), set()
    record_lengths, normal_records, normal_patients = [], set(), set()
    for row in rows:
        record, patient = row['ECG_ID'], row['Patient_ID']
        if not patient or not record or record in ids:
            raise ValueError('Missing patient ID or duplicated recording ID')
        ids.add(record); patients.add(patient)
        statements = [part.split('+') for part in row['AHA_Code'].split(';')]
        if any(code not in terms for part in statements for code in part):
            raise ValueError('Unrecognized diagnostic code')
        primary = {part[0] for part in statements}
        if primary == {'1'}:
            normal_records.add(record); normal_patients.add(patient)
        selected = primary & regions.keys()
        # Independent regex replay checks the statement-boundary parsing.
        replay = set(re.findall(r'(?:^|;)(160|161|165|166)(?=\+|;|$)', row['AHA_Code']))
        if selected != replay:
            raise ValueError('Independent MI statement parser disagrees')
        if selected:
            any_records.add(record); any_patients.add(patient)
        if len(selected) > 1:
            multiregion.add(record)
        for code in selected:
            records[code].add(record); people[code].add(patient)
            modifiers = {modifier for part in statements if part[0] == code for modifier in part[1:]}
            age_modifiers = modifiers & {'330', '331', '332'}
            if not age_modifiers:
                timing[code]['age_unspecified'] += 1
            for modifier in age_modifiers:
                timing[code][terms[modifier]] += 1
        record_lengths.append(int(row['N']) / 500.)
    return dict(checked_utc=datetime.now(timezone.utc).isoformat(), metadata_files=files,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                records=len(rows), patients=len(patients), waveform_rate_hz=500,
                duration_seconds=[min(record_lengths), max(record_lengths)],
                mi_statement_records=len(any_records), mi_statement_patients=len(any_patients),
                multiple_mi_region_statement_records=len(multiregion),
                region_support={code:dict(description=name, records=len(records[code]), patients=len(people[code]),
                                          record_age_modifiers=dict(sorted(timing[code].items())))
                                for code, name in regions.items()},
                normal_ecg_statement_only_records=len(normal_records),
                normal_ecg_statement_only_patients=len(normal_patients),
                independent_parser_agrees=True, waveform_files_downloaded=0, model_fits=0,
                label_mapping_to_target_performed=False, target_patients_added=0,
                limitations=['Metadata screening only; no waveform QC, overlap or identity clearance.',
                             'ECG statements are not independently adjudicated PTB six-class region labels.',
                             'Normal ECG statement does not establish clinical healthy-control status.',
                             'Regional supports may overlap; unspecified MI age is not acute or old.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/raw/open_dataset_screen_20261119/sph'))
    parser.add_argument('--output', type=Path, default=Path('results/open_dataset_screen_20261119/sph_metadata_summary.json'))
    args = parser.parse_args()
    result = screen(args.input)
    if args.output.exists():
        old = json.loads(args.output.read_text())
        result['checked_utc'] = old['checked_utc']
        if old != result:
            raise ValueError('Existing aggregate result changed; preserve it')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
