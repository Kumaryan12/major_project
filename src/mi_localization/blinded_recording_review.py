"""Prepare a local blinded review of archived recording pairs."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from secrets import SystemRandom
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import read_vcg
from .labels import clinical_fields
from .signal import extract_beats


def match_cases(pairs, truth, changed):
    matches = []
    for label in np.unique(truth):
        cases = np.flatnonzero((truth == label) & changed)
        controls = np.flatnonzero((truth == label) & ~changed)
        if not len(cases) or not len(controls):
            continue
        gaps = np.array([pair.days for pair in pairs])
        distances = np.abs(gaps[cases, None] - gaps[controls])
        case_idx, control_idx = linear_sum_assignment(distances)
        matches.extend(zip(cases[case_idx].tolist(), controls[control_idx].tolist()))
    return matches


def plot_case(case_id, pair, raw_root, beats, paper, image_path):
    data = paper['data']
    before, after = data['before_r'], data['after_r']
    time_ms = np.arange(-before, after + 1) * 1000 / data['sampling_rate']
    figure, axes = plt.subplots(3, 2, figsize=(11, 7), sharex=True, sharey='row')
    for column, (record, indices) in enumerate(
            ((pair.earliest, pair.early_rows), (pair.later, pair.later_rows))):
        signals, rate = read_vcg(raw_root, record, data['leads'])
        peaks = beats.loc[indices, 'r_peak'].to_numpy()
        windows, used = extract_beats(signals, peaks, before, after)
        if rate != data['sampling_rate'] or not np.array_equal(peaks, used) or not np.isfinite(windows).all():
            raise ValueError('Raw waveform does not align with retained beat positions')
        representative = np.linspace(0, len(windows) - 1, 5, dtype=int)
        median = np.median(windows, axis=0)
        for lead, axis_name in enumerate('XYZ'):
            axis = axes[lead, column]
            axis.plot(time_ms, windows[representative, lead, :].T, color='#aebbc8', linewidth=.8)
            axis.plot(time_ms, median[lead], color='#163e61', linewidth=1.4)
            axis.axvline(0, color='#666666', linestyle='--', linewidth=.7)
            axis.set_ylabel(f'{axis_name} (mV)')
            axis.grid(alpha=.2)
        axes[0, column].set_title('Earlier recording' if column == 0 else 'Later recording')
        axes[-1, column].set_xlabel('Time from cached R position (ms)')
    figure.suptitle(f'{case_id} · {pair.days} days between recordings')
    figure.text(.5, .007, 'Five evenly spaced raw beats (light) and the median of all retained raw beats (dark)',
                ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .035, 1, .96))
    figure.savefig(image_path, dpi=130)
    plt.close(figure)


def evidence(raw_root, record, record_day):
    header = (raw_root / f'{record}.hea').read_text(encoding='latin1')
    fields = clinical_fields([line for line in header.splitlines() if line.startswith('#')])
    infarct = fields.get('infarction date (acute)', '').strip()
    infarct_days = None
    for fmt in ('%d-%b-%y', '%d-%b-%Y'):
        try:
            infarct_days = (record_day - datetime.strptime(infarct, fmt).date()).days
            break
        except ValueError:
            pass
    vent = ' '.join(fields.get('ventriculography', '').strip().split())
    if re.search(r'\bpatient\s*\d+\b', vent, re.I):
        vent = ''
    return infarct_days, vent or 'Unavailable'


def prepare(output, pairs, classes, truth, earliest_prediction, all_prediction, raw_root, beats, paper):
    if output.exists():
        raise FileExistsError(f'{output} already exists. Keep the original packet and key together.')
    matches = match_cases(pairs, truth, earliest_prediction != all_prediction)
    records = [(i, group, match) for match, (case, control) in enumerate(matches, 1)
               for i, group in ((case, 'changed'), (control, 'stable'))]
    SystemRandom().shuffle(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / 'packet'
        reviewer, custodian = stage / 'reviewer', stage / 'custodian'
        reviewer.mkdir(parents=True, mode=0o700)
        custodian.mkdir(mode=0o700)
        cases = [
            '# Blinded review of measured Frank XYZ recordings', '',
            'Use only the reviewer directory during assessment. Review the waveforms first and record your '
            'observations. Then consult the clinical context and note which regions, if any, have independent '
            'support. Use indeterminate where evidence is limited. Complete the form before unblinding.', '',
            'The plots illustrate retained raw beats on their original axes. The displayed R positions come '
            'from the pipeline. The median summarizes the available beats in each record.', '',
            '[Review form](review_form.md) · [Archival clinical context](clinical_context.md)', '',
        ]
        context = [
            '# Archival clinical context', '',
            'The excerpts below come from the PTB clinical summaries. Time offsets are between documented '
            'calendar dates. Record additional independently verified evidence if available through the custodian.',
        ]
        form = [
            '# Blinded recording review form', '',
            'Reviewer name and date —', '',
            'For each case, describe the quality of each recording, any visible changes, timing context, '
            'regional evidence and your certainty. Record evidence sources and any limitations.', '',
        ]
        key = []
        for number, (i, group, match) in enumerate(records, 1):
            case_id = f'VCG-{number:03d}'
            pair = pairs[i]
            plot_case(case_id, pair, raw_root, beats, paper, reviewer / f'{case_id}.png')
            cases.append(f'- [{case_id}]({case_id}.png)')
            context += ['', f'## {case_id}', '', f'Days between recordings — {pair.days}']
            for slot, record, day in (('Earlier', pair.earliest, pair.earliest_day),
                                      ('Later', pair.later, pair.later_day)):
                infarct_days, vent = evidence(raw_root, record, day)
                interval = 'Unavailable' if infarct_days is None else str(infarct_days)
                context += ['', f'- {slot} recording days after documented infarct date — {interval}',
                            f'- {slot} recording associated ventriculography summary — {vent}']
            form += [f'## {case_id}', '', '- Earlier and later signal quality —',
                     '- Morphology changes across recordings —', '- Timing and clinical context —',
                     '- Region supported by evidence, or indeterminate —', '- Evidence source and certainty —', '']
            key.append(dict(case=case_id, match=match, group=group, patient=pair.patient,
                            records=[pair.earliest, pair.later], source_class=classes[truth[i]],
                            earliest_model_class=classes[earliest_prediction[i]],
                            all_records_model_class=classes[all_prediction[i]]))
        (reviewer / 'START_HERE.md').write_text('\n'.join(cases) + '\n')
        (reviewer / 'clinical_context.md').write_text('\n'.join(context) + '\n')
        (reviewer / 'review_form.md').write_text('\n'.join(form) + '\n')
        (custodian / 'key.json').write_text(json.dumps(key, indent=2) + '\n')
        (custodian / 'START_HERE.md').write_text(
            '# Custodian instructions\n\nKeep this directory private. Share only the reviewer directory '
            'with authorized reviewers. Preserve each completed review before unblinding. The source labels '
            'in the key are the existing PTB annotations.\n')
        stage.rename(output)
    print(f'Prepared {len(matches)} matched review pairs in {output / "reviewer"}')
