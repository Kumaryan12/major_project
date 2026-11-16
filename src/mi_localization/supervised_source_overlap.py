"""Expanded source-used recording screen, without target fits or identity clearance."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.signal import resample_poly
from threadpoolctl import threadpool_limits
import wfdb
import yaml

from . import overlap_screen as numeric, supervised_source_bank as source
from .supervised_source_audit import canonical, file_hash, save_json, safe_path, target_inventory

CLAIM = 'Expanded recording-copy screen only; no patient-identity clearance, PTB classification, localization gain or SOTA claim.'


def progress(config,state,**values):
    save_json(Path(config['output_dir'])/'progress.json',dict(state=state,pid=os.getpid(),
        updated_utc=datetime.now(timezone.utc).isoformat(),**values))


def validate(config):
    numeric.validate_config(config)
    if type(config['candidate_batch_size']) is not int or config['candidate_batch_size']<1:
        raise ValueError('Invalid bounded retrieval batch size')


def completed_pilot(pilot):
    """Inspect completed source artifacts only; never launch preparation or fitting."""
    root=Path(pilot['output_dir']);bank=Path(pilot['bank_dir'])
    required=[root/'summary.json',root/'report_manifest.json',root/'training_manifest.json',bank/'manifest.json']
    if any(not p.is_file() for p in required): raise ValueError('Source pilot is incomplete')
    summary=json.loads((root/'summary.json').read_text())
    report=json.loads((root/'report_manifest.json').read_text())
    training=json.loads((root/'training_manifest.json').read_text())
    if (summary.get('complete') is not True or summary.get('source_only') is not True
            or summary.get('target_evaluated') is not False or training['config']!=pilot
            or report['scope']['training_scope_fingerprint']!=canonical(training)
            or summary['scope_fingerprint']!=canonical(training)
            or training['bank_manifest_sha256']!=file_hash(bank/'manifest.json')):
        raise ValueError('Completed source pilot scope/endpoint mismatch')
    if any(file_hash(root/name)!=digest for name,digest in report['files'].items()):
        raise ValueError('Source pilot public artifact changed')
    expected={(obj,int(seed)) for obj in pilot['objectives'] for seed in pilot['training_seeds']}
    runs=summary['runs']
    if len(runs)!=len(expected) or {(r['objective'],r['seed']) for r in runs}!=expected:
        raise ValueError('Missing declared source objectives/seeds')
    expected_names={f'{obj}_{seed}' for obj,seed in expected}
    if set(report['scope']['run_summary_hashes'])!=expected_names:
        raise ValueError('Source report member coverage changed')
    for r in runs:
        name=f"{r['objective']}_{r['seed']}";member=root/'runs'/name
        if (r['complete'] is not True or r['target_evaluated'] is not False
                or json.loads((member/'summary.json').read_text())!=r
                or file_hash(member/'summary.json')!=report['scope']['run_summary_hashes'][name]
                or any(file_hash(member/file)!=digest for file,digest in r['files'].items())):
            raise ValueError('Source member completion/artifact mismatch')
    code_root=Path(__file__).parent
    for field,name in [('code_sha256','supervised_source_train.py'),('neural_code_sha256','neural.py'),
                       ('report_code_sha256','supervised_source_pilot_report.py')]:
        if training[field]!=file_hash(code_root/name): raise ValueError('Source numerical code changed')
    return summary,dict(report_manifest_sha256=file_hash(root/'report_manifest.json'),
        training_manifest_sha256=file_hash(root/'training_manifest.json'),summary_sha256=file_hash(root/'summary.json'))


def inventory(items):
    result=[]
    for item in items:
        path=Path(item['path']);header=path.with_suffix('.hea');dat=path.with_suffix('.dat')
        h=wfdb.rdheader(str(path))
        names=[str(n).lower() for n in h.sig_name]
        if any(names.count(lead)!=1 for lead in numeric.ECG_LEADS): raise ValueError('Missing/duplicate common ECG leads')
        columns=[names.index(lead) for lead in numeric.ECG_LEADS]
        if (h.fs!=item['sampling_rate'] or h.sig_len<item['sampling_rate']*8
                or any(h.file_name[j]!=dat.name or str(h.units[j]).lower()!='mv'
                       or not np.isfinite(h.adc_gain[j]) or h.adc_gain[j]<=0 for j in columns)):
            raise ValueError('Unsafe/unexpected common-lead source/target header')
        result.append(dict(item,header_sha256=file_hash(header),data_sha256=file_hash(dat)))
    return result


def load_context(config):
    pilot=yaml.safe_load(Path(config['source_config']).read_text())
    summary,evidence=completed_pilot(pilot)
    # Ready-bank path: the preceding guard ensures this cannot start downloads.
    bank=source.prepare(pilot)
    ac=yaml.safe_load(Path(pilot['audit_config']).read_text())
    _,target_counts=target_inventory(ac)
    retained=pd.read_csv(Path(pilot['output_dir'])/'review/retained_records.csv')
    with_ids=np.load(bank/'ecg_ids.npy');groups=np.load(bank/'patient_ids.npy');roles=np.load(bank/'roles.npy')
    if (not retained.ecg_id.is_unique or not np.array_equal(retained.ecg_id.to_numpy(),with_ids)
            or not np.array_equal(retained.patient_id.to_numpy(),groups)
            or not np.array_equal(retained.role.to_numpy(),roles)):
        raise ValueError('Retained source cohort does not match fitted source bank')
    sources=[]
    for row in retained.sort_values('ecg_id').itertuples():
        source.assert_row(row)
        path=safe_path(pilot['waveform_dir'],row.filename_hr)
        for ext in ['.hea','.dat']: safe_path(pilot['waveform_dir'],row.filename_hr+ext)
        sources.append(dict(id=f'ptbxl:ecg:{row.ecg_id}',patient_id=int(row.patient_id),role=row.role,
                            path=str(path),sampling_rate=500))
    beats=pd.read_csv(ac['target_beats_path'],dtype={'patient_id':str})
    beats=beats.loc[beats.label.isin(ac['target_classes'])]
    targets=[]
    for record,part in beats.groupby('record',sort=True):
        path=safe_path(ac['target_waveform_dir'],record)
        for ext in ['.hea','.dat']: safe_path(ac['target_waveform_dir'],record+ext)
        targets.append(dict(id=str(record),patient_id=str(part.patient_id.iloc[0]),path=str(path),sampling_rate=1000))
    if (len(sources)!=summary['bank']['retained_records'] or len(targets)!=target_counts['records']
            or set(retained.role)!={'train','validation'}):
        raise ValueError('Expanded source/target recording coverage mismatch')
    return sources,targets,dict(pilot_evidence=evidence,source_bank_manifest_sha256=file_hash(bank/'manifest.json'),
        source_training_patients=int(retained.loc[retained.role.eq('train'),'patient_id'].nunique()),
        source_validation_patients=int(retained.loc[retained.role.eq('validation'),'patient_id'].nunique()),
        target=target_counts,selected_source_records=summary['bank']['selected_records'])


def replay_cache(item,path,features,available,config):
    """Re-decode every raw common-lead signal for independent cache verification."""
    h=wfdb.rdheader(item['path']);names=[n.lower() for n in h.sig_name]
    r=wfdb.rdrecord(item['path'],channels=[names.index(lead) for lead in numeric.ECG_LEADS])
    if (r.fs!=item['sampling_rate'] or not np.isfinite(r.p_signal).all()
            or any(str(u).lower()!='mv' for u in r.units)):
        raise ValueError('Invalid raw waveform during cache replay')
    factor=int(r.fs)//config['screen_rate']
    if factor*config['screen_rate']!=r.fs: raise ValueError('Invalid screen resampling ratio')
    ecg=resample_poly(r.p_signal,1,factor,axis=0).astype(np.float32)
    rebuilt,present=numeric.templates(ecg,config)
    with np.load(path) as saved:
        if (not np.array_equal(saved['ecg'],ecg) or not np.array_equal(saved['features'],rebuilt)
                or not np.array_equal(features,rebuilt) or available!=present):
            raise ValueError('Raw-decoded screen cache does not reproduce')


def retrieve(sf,tf,sa,ta,count,config):
    candidates=[]
    # Avoid the full (4N) x (4M) similarity matrix for the expanded cohort.
    with threadpool_limits(limits=config['workers']):
        for first in range(0,len(sf),config['candidate_batch_size']):
            last=first+config['candidate_batch_size']
            candidates.extend(numeric.rank_candidates(sf[first:last],tf,sa[first:last],ta,count))
    return candidates


def scan_one(index,indices,tc,sc,config,fingerprint,verify):
    root=Path(config['output_dir'])/'review/pairs';root.mkdir(parents=True,exist_ok=True)
    path=root/f'{index:05d}.json'
    context=dict(fingerprint=fingerprint,target_index=index,source_indices=[int(i) for i in indices])
    if path.exists():
        prior=json.loads(path.read_text())
        if prior['scope']!=context or canonical(prior['rows'])!=prior['rows_sha256']:
            raise ValueError('Expanded pair checkpoint scope/content changed')
        if not verify: return prior['rows']
    elif verify: raise ValueError('Missing expanded pair checkpoint')
    rows=numeric._scan_target(index,indices,tc,sc,config)
    if verify:
        if prior['rows']!=rows: raise ValueError('Continuous pair comparison does not reproduce')
    else: save_json(path,dict(scope=context,rows=rows,rows_sha256=canonical(rows)))
    return rows


def summarize(sources,targets,provenance,rows,controls,count,sa,ta,fingerprint,config):
    pairs=pd.DataFrame(rows);hits=pairs.loc[pairs.candidate]
    hashes={t['data_sha256'] for t in targets}
    matches=[s for s in sources if s['data_sha256'] in hashes]
    incomplete=int((pairs.active_leads<config['minimum_active_leads']).sum())
    controls_ok=sum(bool(r['retrieved_known_record'] and r['confirmed_copied_segment']) for r in controls)
    return dict(complete=True,screen_fingerprint=fingerprint,source_records=len(sources),
        source_patients=len({s['patient_id'] for s in sources}),source_training_patients=provenance['source_training_patients'],
        source_validation_patients=provenance['source_validation_patients'],selected_source_records=provenance['selected_source_records'],
        target_records=len(targets),target_patients=provenance['target']['patients'],retrieval_candidate_count=count,
        continuous_pairs_checked=len(pairs),all_possible_pairs=len(sources)*len(targets),
        positive_controls=len(controls),positive_controls_retrieved_and_confirmed=controls_ok,
        template_unavailable_source_records=int((~sa).sum()),template_unavailable_target_records=int((~ta).sum()),
        incomplete_active_lead_pairs=incomplete,exact_encoded_data_matches=len(matches),near_recording_candidate_pairs=len(hits),
        candidate_source_records=int(hits.source_index.nunique()),maximum_checked_query_mean_correlation=float(pairs.query_mean_correlation.max()),
        recording_screen_passed=bool(count is not None and controls_ok==len(controls) and not matches and hits.empty and incomplete==0),
        patient_identity_independence_cleared=False,target_classification_launched=False,next_experiment_automatically_launched=False,
        coverage='All retained source training and model-selection-validation recordings used by both native-label source models',
        claim_scope=CLAIM,limits='Top-k/fallback retrieval with central eight-second correlation, not exhaustive all-pair all-lag or shorter partial-copy/identity recognition; synthetic copy controls do not establish universal sensitivity')


def report(root,s):
    lines=['# Expanded supervised-source recording-overlap audit','',s['claim_scope'],'',
        f"Covered **{s['source_records']:,} retained source recordings / {s['source_patients']:,} source patients**, including training and source model-selection validation, against **{s['target_records']} PTB recordings / {s['target_patients']} target patients**. The pilot originally selected {s['selected_source_records']:,} records; source-only QC/exact-copy exclusions are inherited from its verified bank. This replaces no historical result and extends no old subset clearance.",'',
        '| Check | Result |','|---|---:|',f"| Encoded-file source–target matches | {s['exact_encoded_data_matches']} |",
        f"| High-correlation candidate pairs | {s['near_recording_candidate_pairs']} |",
        f"| Pairs with fewer than six active leads | {s['incomplete_active_lead_pairs']} |",
        f"| Known-copy controls retrieved and confirmed | {s['positive_controls_retrieved_and_confirmed']}/{s['positive_controls']} |",
        f"| Retrieved/fallback continuous pairs checked | {s['continuous_pairs_checked']:,} / {s['all_possible_pairs']:,} |",
        f"| Recording screen passed | {s['recording_screen_passed']} |",'',
        f"Common physical-mV I/II/V1–V6 ECG is resampled to 250Hz. Median/first/middle/last morphology templates retrieve top-{s['retrieval_candidate_count']} targets per source, escalating through prespecified counts only on synthetic copy controls. Missing source templates use all targets; missing target templates are always included. The source central eight seconds is compared at every valid target lag. At least six active leads, mean correlation≥0.995 and each active lead≥0.98 define a candidate. Full ten-second checks are reported privately when the aligned segment exists. Maximum checked query mean correlation: {s['maximum_checked_query_mean_correlation']:.6f}.",'',
        'Every input file, numerical implementation, runtime, source-bank/source-model provenance and protected target beat/fold checksum is pinned. Private recording caches and pair checkpoints support resume; independent --verify re-decodes every common-lead waveform, repeats retrieval/controls and every selected continuous comparison, and checks all saved artifact hashes. A failed copy-control retrieval/confirmation blocks this screen; any match or insufficient active leads blocks a passed recording-screen flag. Source/target patients are not deleted or relabeled in response to target outcomes.','',
        'This is not a patient-identity clearance: different visits by the same person, shorter copies, transformations outside the controls and missed retrieval remain unresolved. Morphological similarity alone is not proof of duplicate people. The original PTB folds/labels/beats are unchanged, and no target classification predictions, errors or outer scores choose screening thresholds or retrieval budgets.','',
        'No target training or transfer comparison is queued by this audit. After independent verification and aggregate publication, the runner stops. A later matched scratch-versus-supervised-pretrained study would remain exploratory development with unresolved cross-dataset identity, and must be specified separately.','',
        'Protocol: [expanded audit](../../docs/supervised_source_overlap_plan.md). Numerical method: existing project overlap_screen.py; [SciPy correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html).','']
    (root/'REPORT.md').write_text('\n'.join(lines))


def run(config,verify=False):
    validate(config);sources,targets,provenance=load_context(config)
    source_inventory=inventory(sources);target_inventory_list=inventory(targets)
    scope=dict(config=config,provenance=provenance,source_inventory_sha256=canonical(source_inventory),
        target_inventory_sha256=canonical(target_inventory_list),code_sha256=file_hash(__file__),
        numerical_code_sha256=file_hash(numeric.__file__),signal_code_sha256=file_hash(numeric.signal.__file__),
        kors_code_sha256=file_hash(numeric.external.__file__),claim_scope=CLAIM,
        runtime=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,wfdb=wfdb.__version__,platform=platform.platform()))
    fingerprint=canonical(scope);root=Path(config['output_dir']);root.mkdir(parents=True,exist_ok=True)
    review=root/'review';review.mkdir(exist_ok=True);cache=Path(config['cache_dir']);cache.mkdir(parents=True,exist_ok=True)
    guard=root/'screen_manifest.json';scope=dict(scope,screen_fingerprint=fingerprint)
    if guard.exists() and json.loads(guard.read_text())!=scope: raise ValueError('Expanded screen inputs/code/runtime changed; new namespace required')
    if verify and not (root/'output_manifest.json').exists(): raise ValueError('Expanded screen is incomplete')
    if not guard.exists(): save_json(guard,scope)
    finished=(root/'output_manifest.json').exists()
    if finished:
        prior=json.loads((root/'output_manifest.json').read_text())
        if prior['screen_fingerprint']!=fingerprint or any(file_hash(root/name)!=h for name,h in prior['files'].items()):
            raise ValueError('Expanded audit artifact changed')
        if any(file_hash(cache/name)!=h for name,h in prior['cache_files'].items()): raise ValueError('Expanded waveform cache changed')
    if finished and not verify: return json.loads((root/'summary.json').read_text())
    if not finished: save_json(review/'input_provenance.json',dict(source=source_inventory,target=target_inventory_list))
    elif json.loads((review/'input_provenance.json').read_text())!=dict(source=source_inventory,target=target_inventory_list):
        raise ValueError('Expanded input provenance changed')
    items=sources+targets;cached=[None]*len(items)
    def cache_job(i):
        value=numeric._cache_record(items[i],config,fingerprint)
        if verify: replay_cache(items[i],*value,config)
        return i,value
    progress(config,'replaying_waveforms' if verify else 'preparing_screen_waveforms',completed=0,total=len(items))
    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        jobs=[pool.submit(cache_job,i) for i in range(len(items))]
        for n,job in enumerate(as_completed(jobs),1):
            i,value=job.result();cached[i]=value
            if n%100==0 or n==len(jobs): progress(config,'replaying_waveforms' if verify else 'preparing_screen_waveforms',completed=n,total=len(jobs))
    sc=[x[0] for x in cached[:len(sources)]];tc=[x[0] for x in cached[len(sources):]]
    sf=np.stack([x[1] for x in cached[:len(sources)]]);tf=np.stack([x[1] for x in cached[len(sources):]])
    sa=np.asarray([x[2] for x in cached[:len(sources)]],bool);ta=np.asarray([x[2] for x in cached[len(sources):]],bool)
    count,controls=numeric._controls(tc,tf,ta,config)
    if count is None:
        save_json(root/'control_failure.json',dict(complete=False,known_copy_controls_passed=False,classification_launched=False))
        progress(config,'blocked_copy_controls')
        raise ValueError('Known-copy controls failed; expanded screen cannot pass')
    candidates=retrieve(sf,tf,sa,ta,count,config)
    tasks={i:[] for i in range(len(targets))}
    for si,retrieved in enumerate(candidates):
        for ti in retrieved: tasks[int(ti)].append(si)
    controls_path=review/'controls.json';retrieval_path=review/'retrieval.json'
    retrieved_json=[[int(i) for i in values] for values in candidates]
    if finished:
        if json.loads(controls_path.read_text())!=controls or json.loads(retrieval_path.read_text())!=retrieved_json:
            raise ValueError('Expanded controls/retrieval do not reproduce')
    else: save_json(controls_path,controls);save_json(retrieval_path,retrieved_json)
    total=sum(bool(v) for v in tasks.values());rows=[]
    progress(config,'verifying_continuous_pairs' if verify else 'screening_continuous_pairs',completed=0,total=total,
             selected_pairs=sum(map(len,tasks.values())))
    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        jobs=[pool.submit(scan_one,i,ids,tc,sc,config,fingerprint,verify) for i,ids in tasks.items() if ids]
        for n,job in enumerate(as_completed(jobs),1):
            rows.extend(job.result())
            progress(config,'verifying_continuous_pairs' if verify else 'screening_continuous_pairs',completed=n,total=total)
    rows.sort(key=lambda r:(r['target_index'],r['source_index']))
    summary=summarize(source_inventory,target_inventory_list,provenance,rows,controls,count,sa,ta,fingerprint,config)
    # Detect a raw-file change during either build or independent replay.
    if inventory(sources)!=source_inventory or inventory(targets)!=target_inventory_list: raise ValueError('Raw inputs changed during screen')
    if finished:
        if json.loads((root/'summary.json').read_text())!=summary: raise ValueError('Expanded screen counts do not reproduce')
    else:
        save_json(root/'summary.json',summary);report(root,summary)
        files=['summary.json','REPORT.md','review/input_provenance.json','review/controls.json','review/retrieval.json']
        files += [f'review/pairs/{i:05d}.json' for i,ids in tasks.items() if ids]
        save_json(root/'output_manifest.json',dict(screen_fingerprint=fingerprint,files={n:file_hash(root/n) for n in files},
            cache_files={Path(value[0]).name:file_hash(value[0]) for value in cached}))
    progress(config,'independently_verified' if verify else 'screen_complete_pending_independent_replay',
             recording_screen_passed=summary['recording_screen_passed'],target_classification_launched=False)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/supervised_source_overlap_v1.yaml');p.add_argument('--verify',action='store_true')
    args=p.parse_args();print(json.dumps(run(yaml.safe_load(Path(args.config).read_text()),args.verify)))
