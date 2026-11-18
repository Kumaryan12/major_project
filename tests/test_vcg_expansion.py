import csv

import numpy as np
import pytest
import wfdb

from mi_localization import vcg_external_intake as intake
from mi_localization.vcg_expansion_bank import CopySearch
from mi_localization.vcg_pretraining_study import compact_ecg, sampling_plan


def test_copy_screen_recovers_shifted_positive_gain_copy():
    rng = np.random.default_rng(43)
    original = rng.normal(0, .02, size=(1000, 3))
    screen = CopySearch(original, 300)
    config = {'minimum_lead_sd_mv': .001, 'copy_mean_r': .9995, 'copy_min_r': .998}
    shifted = original[50:900] * [1.01, .9, 1.1] + .002
    assert screen.matches(shifted, config)
    assert not screen.matches(rng.normal(0, .02, size=(1000, 3)), config)
    assert not screen.matches(shifted * [1, -1, 1], config)


def test_pretraining_plans_match_exposure_and_shuffled_teacher_marginal():
    config = {'batch_size': 4, 'pretrain_steps': 5, 'native_extra_fraction': .5}
    groups = np.repeat(np.arange(7), 4)
    rows = np.arange(24)
    src = np.repeat(np.arange(10, 13), 3)
    only = sampling_plan(config, 'native', 'target_only', rows, groups, src, 42)
    extra = sampling_plan(config, 'native', 'target_plus_native', rows, groups, src, 42)
    np.testing.assert_array_equal(only[0], extra[0])
    assert only[1].shape == extra[1].shape == (5, 2)
    assert np.isin(only[1], rows).all()
    assert np.all(extra[1] < len(src))

    right = sampling_plan(config, 'bridge', 'paired_ecg', rows, groups, None, 42)
    wrong = sampling_plan(config, 'bridge', 'shuffled_ecg', rows, groups, None, 42)
    np.testing.assert_array_equal(right[0], right[1])
    np.testing.assert_array_equal(right[0], wrong[0])
    np.testing.assert_array_equal(np.sort(right[1], axis=1), np.sort(wrong[1], axis=1))
    assert np.all(groups[wrong[0]] != groups[wrong[1]])
    assert np.isin(wrong[1], rows).all()


def test_ecg_teacher_transform_never_reads_query_rows():
    x = np.arange(6 * 12 * 8, dtype=np.float32).reshape(6, 12, 8)
    training = np.array([0, 1, 3, 4])
    groups = np.array(['a', 'a', 'b', 'c', 'd', 'e'])
    first = compact_ecg(x, training, groups, 3)
    x[[2, 5]] = np.nan
    second = compact_ecg(x, training, groups, 3)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]


def test_partner_intake_checks_real_wfdb_and_rejects_cross_partition_patient(tmp_path, monkeypatch):
    monkeypatch.setattr(intake, 'exact_ptb_hashes', lambda _: ({'patient001'}, set()))
    root = tmp_path / 'signals'
    root.mkdir()
    rng = np.random.default_rng(7)
    for i in range(2):
        wfdb.wrsamp(f'exam{i}', fs=1000, units=['mV'] * 3, sig_name=['vx', 'vy', 'vz'],
                    p_signal=rng.normal(0, .02, size=(700, 3)), fmt=['16'] * 3, write_dir=str(root))
    manifest = tmp_path / 'partner.csv'
    rows = [
        ['exam0', 'pseudo1', 's1', 'external_test', 'AMI', 'confirmed', 'cardiac_mri'],
        ['exam1', 'pseudo2', 's1', 'development', '', 'indeterminate', ''],
    ]
    receipt = tmp_path / 'summary.json'

    def save():
        with manifest.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(intake.FIELDS)
            writer.writerows(rows)

    save()
    result = intake.validate(manifest, root, root, receipt)
    assert result['confirmed_patients_by_partition_and_class']['external_test']['AMI'] == 1
    assert result['patients_by_review_status'] == {'confirmed': 1, 'indeterminate': 1}
    assert not result['external_has_all_classes']
    assert intake.validate(manifest, root, root, receipt) == result
    rows[1][1] = 'pseudo1'
    save()
    with pytest.raises(ValueError, match='inconsistent'):
        intake.validate(manifest, root, root, receipt)


def test_external_evaluation_gates_on_every_verified_fold_seed_checkpoint(tmp_path):
    import json

    from mi_localization import vcg_external_evaluation as evaluation

    def save(path, value):
        path.write_text(json.dumps(value))
        return evaluation.file_hash(path)

    arm, fingerprint = 'target_plus_native', 'test-scope'
    seeds = [42, 123, 2026]
    save(tmp_path / 'study_manifest.json', dict(
        fingerprint=fingerprint, smoke=False, code={},
        config=dict(classes=intake.CLASSES, outer_splits=10, training_seeds=seeds)))
    checks = {}
    for fold in range(1, 11):
        for seed in seeds:
            name = f'fold{fold:02d}_seed{seed}_{arm}'
            directory = tmp_path / 'review' / name
            directory.mkdir(parents=True)
            model = directory / 'model.pt'
            model.write_bytes(f'{fold}-{seed}'.encode())
            checks[name] = save(directory / 'audit.json', dict(
                context=dict(fingerprint=fingerprint, arm=arm, fold=fold, declared_seed=seed),
                files={'model.pt': evaluation.file_hash(model)}))
    verified = save(tmp_path / 'verification_manifest.json', dict(fingerprint=fingerprint, fit_audit_hashes=checks))
    report = tmp_path / 'REPORT.md'
    report.write_text('Synthetic verification receipt')
    save(tmp_path / 'summary.json', dict(
        complete=True, study_fingerprint=fingerprint, ensemble_metrics={arm: {}},
        verification_manifest_sha256=verified, report_sha256=evaluation.file_hash(report)))
    _, members = evaluation.checked_members(tmp_path, arm)
    assert len(members) == 30
    members[-1][0].write_bytes(b'modified')
    with pytest.raises(ValueError, match='membership changed'):
        evaluation.checked_members(tmp_path, arm)
