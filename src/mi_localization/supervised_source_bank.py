"""Bounded native-label PTB-XL bank in a new cache; no target reads or evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil

import numpy as np
import pandas as pd
import scipy
from scipy.signal import resample_poly
import wfdb
import yaml

from . import data, external, signal, supervised_source_audit as audit

CLAIM = "Source-only native multilabel diagnostic pilot on Kors-derived XYZ; no PTB evaluation, clinical negative-label adjudication or cross-dataset identity clearance."


def progress(config, state, **values):
    audit.save_json(Path(config['output_dir'])/'progress.json', dict(
        state=state, pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(), **values))


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def source_metadata(config):
    ac = yaml.safe_load(Path(config['audit_config']).read_text())
    audit.validate_config(ac)
    for name in ['metadata', 'statements']:
        if audit.file_hash(ac[f'{name}_path']) != ac[f'{name}_sha256']:
            raise ValueError('Pinned official source metadata changed')
    if (type(config['training_patients']) is not int or config['training_patients'] < 2
            or type(config['cohort_seed']) is not int or config['cohort_seed'] < 0):
        raise ValueError('Invalid source cohort budget/seed')
    for name in ['beats_per_record', 'download_workers', 'conversion_workers', 'minimum_training_positive_patients']:
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError('Invalid source bank setting')
    settings = config['preprocessing']
    if (settings['sampling_rate'], settings['before_r'], settings['after_r']) != (1000, 250, 400):
        raise ValueError('Require original 1000 Hz / 651 sample crop')
    if (Path(config['waveform_dir']).resolve() == Path(config['reuse_waveform_dir']).resolve()
            or Path(config['waveform_dir']).resolve().is_relative_to(Path(config['reuse_waveform_dir']).resolve())):
        raise ValueError('Use a separate expanded waveform cache; preserve the old audit inventory')
    if config['base_url'] != 'https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3':
        raise ValueError('Require the versioned official source mirror')
    frame = audit.source_frame(pd.read_csv(ac['metadata_path']), ac)
    statements = pd.read_csv(ac['statements_path'], index_col=0)
    labels, _, states, likelihood = audit.label_states(frame, statements, ac)
    return ac, frame, labels, states, likelihood


def select_cohort(frame, labels, states, config):
    """All train patients with >=1 maximum-likelihood review MI code, then random fill.

    Validation fold is retained in full; its annotations never choose training patients.
    All selected patients' visits remain present, with record-specific annotations.
    """
    train = frame.role.eq('train').to_numpy()
    columns = [labels.index(k) for k in ['AMI', 'ALMI', 'ASMI', 'IMI', 'ILMI']]
    mandatory = np.sort(frame.loc[train & np.any(states[:, columns] == 3, axis=1), 'patient_id'].unique())
    candidates = np.sort(frame.loc[train, 'patient_id'].unique())
    budget = config['training_patients']
    if len(mandatory) > budget or budget > len(candidates):
        raise ValueError('Training budget must contain all prespecified MI-positive patients and fit source population')
    remaining = np.setdiff1d(candidates, mandatory)
    fill = np.random.default_rng(config['cohort_seed']).permutation(remaining)[:budget-len(mandatory)]
    selected = frame.patient_id.isin(np.r_[mandatory, fill]) | frame.role.eq('validation')
    cohort = frame.loc[selected].copy().reset_index(drop=True)
    return cohort, np.flatnonzero(selected.to_numpy()), len(mandatory)


def assert_row(row):
    if ((row.role == 'train' and row.strat_fold not in range(1, 9))
            or (row.role == 'validation' and row.strat_fold != 9)
            or row.role not in {'train', 'validation'} or row.strat_fold == 10):
        raise ValueError('Refusing excluded/wrong-role source waveform access')
    import re
    if (not re.fullmatch(r'records500/\d{5}/\d{5}_hr', str(row.filename_hr))
            or int(str(row.filename_hr).rsplit('/', 1)[1].split('_')[0]) != int(row.ecg_id)):
        raise ValueError('Unsafe/mismatched source waveform path')


def freeze(config):
    ac, frame, labels, states, likelihood = source_metadata(config)
    cohort, indices, mandatory = select_cohort(frame, labels, states, config)
    content = cohort[['ecg_id', 'patient_id', 'strat_fold', 'role', 'filename_hr', 'scp_codes']].to_csv(index=False).encode()
    scope = dict(config=config, audit_config=ac, cohort_sha256=hashlib.sha256(content).hexdigest(),
        code_sha256=audit.file_hash(__file__), audit_code_sha256=audit.file_hash(audit.__file__),
        external_code_sha256=audit.file_hash(external.__file__), signal_code_sha256=audit.file_hash(signal.__file__),
        download_code_sha256=audit.file_hash(data.__file__), kors_matrix=external.KORS_MATRIX.tolist(),
        runtime=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,wfdb=wfdb.__version__,platform=platform.platform()),
        claim_scope=CLAIM)
    scope['fingerprint'] = audit.canonical(scope)
    root = Path(config['output_dir']);root.mkdir(parents=True, exist_ok=True)
    review = root/'review';review.mkdir(exist_ok=True)
    guard = root/'cohort_manifest.json'
    if guard.exists() and json.loads(guard.read_text()) != scope:
        raise ValueError('Frozen supervised source scope changed; use a new namespace')
    if (review/'cohort.csv').exists() and (review/'cohort.csv').read_bytes() != content:
        raise ValueError('Frozen source cohort changed')
    audit.save_json(guard, scope)
    (review/'cohort.csv').write_bytes(content)
    audit.save_json(root/'cohort_summary.json', dict(
        fingerprint=scope['fingerprint'], mandatory_mi_training_patients=mandatory,
        roles={r:audit.role_counts(cohort,cohort.role.eq(r)) for r in ['train','validation']},
        validation_annotation_used_for_training_selection=False, excluded_fold_10_records=0,
        all_selected_patient_visits_retained=True, target_evaluated=False, claim_scope=CLAIM))
    return cohort, labels, states[indices], likelihood[indices], scope


def download_row(row, config):
    assert_row(row)
    for suffix in ['hea', 'dat']:
        relative = f'{row.filename_hr}.{suffix}'
        dest = audit.safe_path(config['waveform_dir'], relative)
        if not dest.exists():
            old = audit.safe_path(config['reuse_waveform_dir'], relative)
            if old.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                temporary = dest.with_suffix(dest.suffix+'.partial')
                shutil.copyfile(old, temporary);temporary.replace(dest)
            else:
                data.download_file(config['base_url']+'/'+relative, dest)
    header = audit.safe_path(config['waveform_dir'], f'{row.filename_hr}.hea')
    dat = header.with_suffix('.dat')
    if audit.inspect_header(header, dat.name) != 'compatible_header' or dat.stat().st_size != 120000:
        raise ValueError('Source download has incompatible header/encoding/size')


def convert(row, config, fingerprint):
    assert_row(row)
    path = audit.safe_path(config['waveform_dir'], str(row.filename_hr))
    hashes = {ext:audit.file_hash(path.with_suffix('.'+ext)) for ext in ['hea','dat']}
    part = Path(config['bank_dir'])/'parts'/f'{row.ecg_id:05d}.npz'
    if part.exists():
        with np.load(part) as saved:
            item = json.loads(str(saved['audit'].item()));waves=saved['waves'];peaks=saved['peaks']
            if (item['fingerprint'] != fingerprint or item['raw_hashes'] != hashes
                    or item['ecg_id'] != int(row.ecg_id) or item['patient_id'] != int(row.patient_id)
                    or item['role'] != row.role or item['waves_sha256'] != array_hash(waves)
                    or item['peaks_sha256'] != array_hash(peaks)):
                raise ValueError('Source conversion cache changed')
            return waves, peaks, item
    if audit.inspect_header(path.with_suffix('.hea'), path.name+'.dat') != 'compatible_header':
        raise ValueError('Incompatible source header before decoding')
    record = wfdb.rdrecord(str(path))
    if record.fs != 500 or record.p_signal.shape != (5000,12):
        raise ValueError('Unexpected decoded source shape/rate')
    item = dict(ecg_id=int(row.ecg_id),patient_id=int(row.patient_id),role=row.role,
                fingerprint=fingerprint,raw_hashes=hashes,accepted=False,reason='nonfinite_ecg')
    waves = np.empty((0,3,651),np.float32);peaks = np.empty(0,np.int64)
    if np.isfinite(record.p_signal).all():
        xyz = resample_poly(external.kors_transform(record.p_signal,record.sig_name),2,1,axis=0)
        item['physical_xyz_sha256'] = array_hash(xyz.astype('<f4'))
        item['xyz_rms_mv'] = np.sqrt(np.mean(xyz**2,axis=0)).tolist()
        item['xyz_max_abs_mv'] = np.max(np.abs(xyz),axis=0).tolist()
        item['reason'] = 'nonfinite_or_flat_xyz'
        if np.isfinite(xyz).all() and np.linalg.norm(xyz-xyz.mean(0)) > 1e-8:
            s=config['preprocessing']
            clean=signal.denoise_vcg(xyz,s['wavelet'],s['denoise_level'])
            detected=signal.detect_r_peaks(clean,1000,s['rpeak_refractory_ms'],s['rpeak_integration_ms'],s['rpeak_threshold_scale'])
            valid=detected[(detected>=250)&(detected+400<len(clean))]
            item['complete_detected_crops']=len(valid);item['reason']='no_complete_detected_crop'
            if len(valid):
                peaks=valid[np.linspace(0,len(valid)-1,min(config['beats_per_record'],len(valid)),dtype=int)].astype(np.int64)
                waves=np.stack([clean[p-250:p+401].T for p in peaks]).astype(np.float32)
                item['accepted']=bool(np.isfinite(waves).all() and np.linalg.norm(waves-waves.mean(2,keepdims=True))>1e-8)
                item['reason']='accepted' if item['accepted'] else 'invalid_or_flat_crop'
                if not item['accepted']: waves=np.empty((0,3,651),np.float32);peaks=np.empty(0,np.int64)
    item['waves_sha256']=array_hash(waves);item['peaks_sha256']=array_hash(peaks)
    part.parent.mkdir(parents=True, exist_ok=True)
    temporary=part.with_suffix('.partial.npz')
    np.savez_compressed(temporary,waves=waves,peaks=peaks,audit=json.dumps(item));temporary.replace(part)
    return waves,peaks,item


def quarantine(records):
    """All members of cross-patient copies; deterministic same-patient deduplication."""
    records=records.copy();groups=[];cross=set();same=set()
    for key in ['data_sha256','physical_xyz_sha256']:
        if key not in records: continue
        eligible=records.loc[records[key].notna() & records[key].ne('')]
        for digest, part in eligible.groupby(key):
            if len(part)<2: continue
            different=part.patient_id.nunique()>1
            groups.append(dict(kind=key,sha256=digest,records=part.ecg_id.astype(int).tolist(),cross_patient=different,
                               cross_role=part.role.nunique()>1))
            if different: cross.update(part.ecg_id)
            else: same.update(part.sort_values('ecg_id').ecg_id.iloc[1:])
    records.loc[records.ecg_id.isin(same),['accepted','reason']]=[False,'same_patient_exact_duplicate']
    records.loc[records.ecg_id.isin(cross),['accepted','reason']]=[False,'cross_patient_exact_duplicate']
    return records,groups


def prepare(config):
    cohort,labels,states,likelihood,scope=freeze(config)
    bank=Path(config['bank_dir']);bank.mkdir(parents=True,exist_ok=True)
    root=Path(config['output_dir']);manifest=bank/'manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['fingerprint'] != scope['fingerprint'] or any(audit.file_hash(bank/name)!=h for name,h in saved['files'].items()):
            raise ValueError('Supervised source bank changed')
        prior=pd.read_csv(root/'review/record_qc.csv').set_index('ecg_id')
        if audit.file_hash(root/'review/record_qc.csv') != saved['record_qc_sha256']:
            raise ValueError('Source QC audit changed')
        for relative,key in [('bank_summary.json','bank_summary_sha256'),
                             ('review/duplicate_groups.json','duplicate_groups_sha256'),
                             ('review/retained_records.csv','retained_records_sha256')]:
            if audit.file_hash(root/relative)!=saved[key]:
                raise ValueError('Source bank report/review content changed')
        for row in cohort.itertuples():
            assert_row(row)
            for ext in ['hea','dat']:
                path=audit.safe_path(config['waveform_dir'],f'{row.filename_hr}.{ext}')
                if audit.file_hash(path)!=prior.loc[row.ecg_id,ext+'_sha256']:
                    raise ValueError('Source raw waveform changed')
        return bank
    progress(config,'downloading_source_only',completed=0,total=len(cohort))
    with ThreadPoolExecutor(max_workers=config['download_workers']) as pool:
        jobs=[pool.submit(download_row,row,config) for row in cohort.itertuples()]
        for n,job in enumerate(as_completed(jobs),1):
            job.result()
            if n%100==0 or n==len(jobs):
                progress(config,'downloading_source_only',completed=n,total=len(jobs))
                print(f'Source download/verification {n}/{len(jobs)}',flush=True)
    converted={}
    with ThreadPoolExecutor(max_workers=config['conversion_workers']) as pool:
        jobs={pool.submit(convert,row,config,scope['fingerprint']):int(row.ecg_id) for row in cohort.itertuples()}
        for n,job in enumerate(as_completed(jobs),1):
            converted[jobs[job]]=job.result()
            if n%100==0 or n==len(jobs):
                progress(config,'converting_source_only',completed=n,total=len(jobs))
                print(f'Source XYZ conversion {n}/{len(jobs)}',flush=True)
    items=[]
    for row in cohort.itertuples():
        item=dict(converted[int(row.ecg_id)][2]);hashes=item.pop('raw_hashes')
        item.update(hea_sha256=hashes['hea'],dat_sha256=hashes['dat'],data_sha256=hashes['dat'])
        items.append(item)
    records,duplicates=quarantine(pd.DataFrame(items))
    records.to_csv(root/'review/record_qc.csv',index=False)
    audit.save_json(root/'review/duplicate_groups.json',duplicates)
    keep=np.flatnonzero(records.accepted.to_numpy());retained=cohort.iloc[keep].reset_index(drop=True)
    if set(retained.role) != {'train','validation'} or set(retained.loc[retained.role.eq('train'),'patient_id']) & set(retained.loc[retained.role.eq('validation'),'patient_id']):
        raise ValueError('Invalid retained source roles')
    values=np.zeros((len(keep),config['beats_per_record'],3,651),np.float32)
    counts=np.zeros(len(keep),np.int64);peak_array=np.full((len(keep),config['beats_per_record']),-1,np.int64)
    for i,row in retained.iterrows():
        waves,peaks,_=converted[int(row.ecg_id)];values[i,:len(waves)]=waves;counts[i]=len(waves);peak_array[i,:len(peaks)]=peaks
    arrays=dict(waves=values,counts=counts,peaks=peak_array,states=states[keep],likelihood=likelihood[keep],
                labels=np.asarray(labels,dtype=str),patient_ids=retained.patient_id.to_numpy(np.int64),
                roles=retained.role.to_numpy(dtype=str),ecg_ids=retained.ecg_id.to_numpy(np.int64))
    for name,value in arrays.items():
        temporary=bank/(name+'.partial.npy');np.save(temporary,value);temporary.replace(bank/(name+'.npy'))
    retained.to_csv(root/'review/retained_records.csv',index=False)
    summary=dict(complete=True, fingerprint=scope['fingerprint'], selected_records=len(cohort),retained_records=len(retained),
        roles={r:audit.role_counts(retained,retained.role.eq(r)) for r in ['train','validation']},
        exclusions={str(k):int(v) for k,v in records.loc[~records.accepted].reason.value_counts().items()},
        duplicate_groups=len(duplicates),cross_patient_duplicate_groups=sum(d['cross_patient'] for d in duplicates),
        physiological_checks='Finite/nonflat derived XYZ, complete detected crops; no expert beat/quality adjudication',
        target_evaluated=False,cross_dataset_identity_cleared=False,fold_10_waveform_accessed=False,claim_scope=CLAIM)
    audit.save_json(root/'bank_summary.json',summary)
    audit.save_json(manifest,dict(fingerprint=scope['fingerprint'],scope=scope,files={name+'.npy':audit.file_hash(bank/(name+'.npy')) for name in arrays},
        record_qc_sha256=audit.file_hash(root/'review/record_qc.csv'),bank_summary_sha256=audit.file_hash(root/'bank_summary.json'),
        duplicate_groups_sha256=audit.file_hash(root/'review/duplicate_groups.json'),
        retained_records_sha256=audit.file_hash(root/'review/retained_records.csv')))
    progress(config,'source_bank_ready',retained_records=len(retained))
    return bank


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/supervised_source_pilot_v1.yaml')
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    print(prepare(config) if args.prepare else freeze(config)[-1]['fingerprint'])
