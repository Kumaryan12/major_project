"""Aggregate source-only feasibility results; no target score or automatic transfer."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from .supervised_source_audit import canonical,file_hash,save_json


PUBLIC_FILES=['summary.json','REPORT.md','metrics_by_code.csv','training_history.csv','learning_curves.png']


def render(config,results,scope,verify=False):
    root=Path(config['output_dir'])
    bank=json.loads((root/'bank_summary.json').read_text())
    cohort=json.loads((root/'cohort_summary.json').read_text())
    summary=dict(complete=True,source_only=True,target_evaluated=False,cross_dataset_identity_cleared=False,
        scope_fingerprint=canonical(scope),cohort=cohort,bank=bank,runs=results,
        comparison='Both declared objectives/all declared seeds on the same native source validation patients; no independent test or PTB transfer',
        endpoint='Patient-weighted record-annotation macro average precision on common likelihood100/unlisted masks',
        clinical_negatives=False,clinical_source_target_equivalence_approved=False,
        next_stage_automatically_launched=False)
    report_scope=dict(training_scope_fingerprint=canonical(scope),
        run_summary_hashes={f"{r['objective']}_{r['seed']}":file_hash(root/'runs'/f"{r['objective']}_{r['seed']}"/'summary.json') for r in results})
    manifest=root/'report_manifest.json'
    if verify or manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['scope']!=report_scope or any(file_hash(root/name)!=digest for name,digest in saved['files'].items()):
            raise ValueError('Source report artifacts/scope changed')
        if json.loads((root/'summary.json').read_text())!=summary:
            raise ValueError('Source report counts/metrics do not reproduce')
        return summary
    save_json(root/'summary.json',summary)
    pd.DataFrame([dict(objective=r['objective'],seed=r['seed'],**c) for r in results for c in r['metrics']['per_code']]).to_csv(root/'metrics_by_code.csv',index=False)
    pd.DataFrame([dict(objective=r['objective'],seed=r['seed'],**e) for r in results for e in r['history']]).to_csv(root/'training_history.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for i,r in enumerate(results):
        ax.plot([0]+[e['epoch'] for e in r['history']],
                [r['initial_metrics']['macro_average_precision']]+[e['validation_macro_ap'] for e in r['history']],
                label=f"{r['objective']}, seed {r['seed']}",color=str(.15+.5*(i%2)),linestyle='-' if i%2==0 else '--')
    ax.axhline(results[0]['training_prior_baseline']['macro_average_precision'],color='.5',linestyle=':',label='Training-prevalence constant predictor')
    ax.set(xlabel='Source training epoch (0 = initialization)',ylabel='Patient-weighted validation macro AP',title='Source-only native diagnostic feasibility')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=160);plt.close(fig)
    lines=['# Supervised native-annotation source pilot','',
        'Completed source-only training and restored-checkpoint verification. This is multilabel diagnostic learning on ECG-derived XYZ, not the six-class measured-VCG localization endpoint. No target training/evaluation or follow-on transfer was launched.','',
        f"Selected {cohort['roles']['train']['patients']:,} training patients / {cohort['roles']['train']['records']:,} records, including all {cohort['mandatory_mi_training_patients']:,} native training patients with at least one likelihood100 review MI code. Fill patients were sampled deterministically without validation labels. All selected visits were retained before QC. Full native fold9 validation: {cohort['roles']['validation']['patients']:,} patients / {cohort['roles']['validation']['records']:,} records. Fold10 was excluded before label/signal access.",'',
        '| Role | Retained patients | Retained records |','|---|---:|---:|']
    for role in ['train','validation']:
        b=bank['roles'][role];lines.append(f"| {role} | {b['patients']} | {b['records']} |")
    lines+=['',f"QC/exact-copy exclusions: `{json.dumps(bank['exclusions'])}`. Detected exact duplicate groups: {bank['duplicate_groups']} (raw-encoded and physical-XYZ grouping may overlap). Every cross-patient group member is quarantined; same-patient copies retain the lowest recording ID. No near-copy screen, clinical signal-quality adjudication or cross-dataset identity clearance is claimed.",'',
        f"Training heads: {len(scope['labels'])} native codes with at least {config['minimum_training_positive_patients']} distinct retained training patients at likelihood100. Labels: {', '.join(scope['labels'])}. Validation support never chooses heads. Native normal ECG (NORM) is not automatically the target clinical HC definition; broad AMI parents are not the exclusive target AMI class.",'',
        '## Declared annotation objectives','',
        'Both objectives treat unlisted codes as annotation negatives for this research benchmark. This is a declared supervision assumption, not adjudicated disease absence. Listed likelihood0 is unknown and masked in both. The primary certain objective trains only listed100 positives plus unlisted negatives, masking intermediate annotations. The positive_intermediate sensitivity also treats listed intermediate annotations as positive; likelihood values are not calibrated soft probabilities. Both arms use identical common validation masks: listed100 positives and unlisted annotation negatives; unknown/intermediate entries are excluded.','',
        'The primary source selection metric is patient-weighted record macro average precision (AP). Records carry weight 1/(retained records for that patient); labels remain record-specific, with no union across visits. Codes without both positive and negative validation records are reported but omitted from macro metrics. F1 uses a fixed 0.5 threshold, with no validation threshold search. This source F1 is not the project six-class patient macro F1.','',
        '| Objective | Seed | Best epoch / trained epochs | Macro AP | Macro AUROC | Macro F1 at 0.5 | Evaluable / trained codes |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        m=r['metrics'];lines.append(f"| {r['objective']} | {r['seed']} | {r['best_epoch']} / {r['epochs_completed']} | {m['macro_average_precision']:.6f} | {m['macro_auroc']:.6f} | {m['macro_f1_at_05']:.6f} | {m['evaluable_codes']} / {m['total_trained_codes']} |")
    baseline=results[0]['training_prior_baseline']
    lines+=['',f"Training-prevalence constant baseline: macro AP {baseline['macro_average_precision']:.6f}, AUROC {baseline['macro_auroc']:.6f}, F1@0.5 {baseline['macro_f1_at_05']:.6f}. Each run also reports its random-initialization control in summary.json. Both objectives and all declared seeds are retained; no winner replaces the declared comparison.",'',
        '![Source validation learning curves](learning_curves.png)','',
        '## Representation, fitting and verification','',
        f"Each recording is transformed from standard 12-lead ECG to signed Kors XYZ, polyphase-resampled 500→1000Hz, denoised with existing bior6.8 settings, and represented by up to {config['beats_per_record']} uniformly spaced complete detector-centered 651-sample crops. Fewer crops use explicit padding masks. Detector positions and infarct boundaries are not expert-adjudicated. The existing small VCG ResNet encoder is shared across crops; their features are averaged within the recording before a native multilabel head. This is weak record-level supervision, not beat-level clinical labels or a new temporal architecture.",'',
        f"Normalization uses only training crops, giving each patient equal total record weight and each record equal crop weight. Patient-uniform record sampling uses {config['samples_per_epoch']:,} presentations per epoch with replacement. Positive loss weights come only from training annotations, capped at {config['maximum_positive_weight']}. AdamW, gradient clipping, at most {config['max_epochs']} epochs, patience {config['patience']}, and epoch-zero fallback were declared before training. The same initialization/sampler policy is used across the two objectives for each seed. Best checkpoints use common validation AP, so this validation is model-selection data, not independent confirmation. MPS is seeded but not guaranteed bitwise deterministic.",'',
        'Checkpoint, optimizer, best state and sampler/loader/torch/device RNG states permit compatible interruption recovery. Each completed member is restored and its full validation probabilities recomputed, alongside scopes, normalization and file hashes. Local encoder exports carry explicit source-only and unresolved-identity flags. Private weights, source record labels, crop caches and recording predictions remain ignored; aggregate metrics/history and manifests are public.','',
        '## Interpretation and stopping boundary','',
        'Diagnostic source learning can establish whether the signal contains learnable native annotations under the declared convention. It does not demonstrate improved PTB localization, clinical validity, clean patient-independent cross-dataset transfer, a 0.80–0.90 target score or SOTA. Training enrichment changes source prevalence; the full validation fold retains its native prevalence. One declared training seed is a bounded feasibility check, not repeatability evidence.','',
        'Stop here. A later part must separately specify source waveform/provenance overlap checks and matched scratch-versus-supervised-pretrained target evaluation inside the existing patient partitions. The original 164-patient target and frozen ridge/neural development references are unchanged.','',
        'Protocol: [supervised source pilot](../../docs/supervised_source_pilot_plan.md). Source documentation: [PTB-XL1.0.3](https://physionet.org/content/ptb-xl/1.0.3/).','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    save_json(manifest,dict(scope=report_scope,files={name:file_hash(root/name) for name in PUBLIC_FILES}))
    return summary
