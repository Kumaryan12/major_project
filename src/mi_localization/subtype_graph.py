"""Nested graph regularization of six patient-level convolution ridge scores."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import softmax
from threadpoolctl import threadpool_limits
import torch
import yaml

from . import benchmark, distribution_kernel_source as source, distribution_kernel_model as kernel
from . import distribution_score_correction as artifacts, distribution_study as core
from .multiview_bank import array_hash
from .neural import _hash, _save_json
from .signal_audit import file_hash
from .subtype_graph_report import summarize

ARMS = ('ridge', 'subtype_graph', 'relabeled_graph')


def laplacian(classes, edges):
    index = {name: i for i, name in enumerate(classes)}
    result = np.zeros((len(classes), len(classes)))
    known = set()
    for left, right in edges:
        if left == right or 'HC' in (left, right) or left not in index or right not in index:
            raise ValueError('Edges must connect distinct declared MI subtypes')
        edge = tuple(sorted((left, right)))
        if edge in known:
            raise ValueError('Duplicate graph edge')
        known.add(edge)
        i, j = index[left], index[right]
        result[i, j] = result[j, i] = -1
        result[i, i] += 1
        result[j, j] += 1
    return result


def make_graphs(config):
    classes = config['classes']
    strength = config['strengths']
    if (tuple(config['graphs']) != ARMS[1:] or len(strength) != 3 or
            not np.isfinite(strength).all() or not 0 == strength[0] < strength[1] < strength[2]):
        raise ValueError('Require matched graph arms with off and two positive strengths')
    graphs = {arm: laplacian(classes, config['graphs'][arm]) for arm in ARMS[1:]}
    first, second = (graphs[arm] for arm in ARMS[1:])
    if (not np.any(first) or np.array_equal(first, second)
            or not np.allclose(np.linalg.eigvalsh(first), np.linalg.eigvalsh(second))):
        raise ValueError('The graph control must differ while preserving the graph spectrum')
    return graphs


def fit_dual(data, alpha, strength, graph):
    """Solve weighted graph ridge in the six eigenvectors of the output penalty."""
    gram, _, _ = data.kernel(0)
    weights = np.sqrt(data.weights)
    weighted = (gram * weights[:, None]) * weights[None, :]
    eigenvalues, basis = np.linalg.eigh(graph)
    if min(eigenvalues) < -1e-12 or alpha <= 0 or strength < 0:
        raise ValueError('Invalid graph ridge penalty')
    targets = (weights[:, None] * data.y_centered) @ basis
    latent = np.empty_like(targets)
    for i, value in enumerate(eigenvalues):
        penalty = alpha * (1 + strength * max(0., value))
        system = weighted.copy()
        system.flat[::len(system) + 1] += penalty
        latent[:, i] = cho_solve(cho_factor(system, lower=True), targets[:, i])
    coefficients = (weights[:, None] * latent) @ basis.T
    if not np.isfinite(coefficients).all():
        raise ValueError('Nonfinite graph ridge coefficients')
    return coefficients


@dataclass
class Context:
    data: kernel.KernelData
    query: np.ndarray
    inner: int
    base: list
    base_scores: list
    audit: dict


def prepare_context(bank, scope, seed, fold, inner, train, query, candidate_ids):
    config = scope['config']
    features, source_path = source.context(bank, seed, fold, inner, train, query)
    values = features['baseline']
    if values.shape != (len(train) + len(query), bank.summary['base_dimensions']):
        raise ValueError('Expected complete baseline feature block')
    n_classes = len(config['classes'])
    data = kernel.KernelData(values[:len(train)], bank.labels[train], values[len(train):], n_classes)
    probabilities, scores, references = [], [], {}
    for i in candidate_ids:
        p, saved, path = source.reference(bank, seed, fold, inner, 'baseline', i,
                                          train, query, values, source_path)
        reference = kernel.KernelModel.fit(data, config['ridge_alphas'][i], 0)
        replay = reference.predict(data)
        if (not np.allclose(replay, p, rtol=1e-8, atol=1e-10)
                or not np.array_equal(replay.argmax(1), p.argmax(1))):
            raise ValueError('Independent ridge solve does not match the source')
        references[str(i)] = file_hash(path)
        probabilities.append(p)
        if inner:
            score = artifacts.score(bank.labels[query], p, n_classes)
            if score != saved['validation_macro_f1']:
                raise ValueError('Original inner ridge score changed')
            scores.append(score)
    audit = dict(core.membership(bank, scope, train, query), outer_fold=fold, seed=seed, inner_fold=inner,
                 source_context_sha256=file_hash(source_path), baseline_audits=references,
                 kernel_state=data.state, baseline_inner_scores=scores)
    return Context(data, query, inner, probabilities, scores, audit)


def graph_context(bank, scope, output, context, graphs, candidate, verify, selected=None):
    config = scope['config']
    data = context.data
    alpha, strengths = config['ridge_alphas'][candidate], config['strengths']
    base = context.base[candidate] if context.inner else context.base[0]
    _, cross, _ = data.kernel(0)
    arrays = {'query_rows': context.query, 'ridge': base}
    scores = {}
    for arm, graph in graphs.items():
        indices = range(len(strengths)) if context.inner else [selected[arm]]
        predictions = []
        for index in indices:
            if index == 0:
                dual = np.empty((0, len(config['classes'])))
                p = base
            else:
                dual = fit_dual(data, alpha, strengths[index], graph)
                p = softmax(cross @ dual + data.y_mean, axis=1)
            predictions.append(p)
            arrays[f'{arm}_dual_{index}'] = dual
        arrays[arm] = np.stack(predictions)
        if context.inner:
            scores[arm] = [artifacts.score(bank.labels[context.query], p, len(config['classes']))
                           for p in predictions]
    audit = dict(context.audit, baseline_candidate=candidate, alpha=alpha, inner_scores=scores,
                 selected_strength_indices=selected, intercept_sha256=array_hash(data.y_mean))
    path = output / f"fold{audit['outer_fold']:02d}_s{audit['seed']}_inner{context.inner}.json"
    restored = artifacts.save_record(path, audit, arrays, verify)
    if context.inner:
        return scores
    result = {arm: restored[arm][0] for arm in graphs}
    result['ridge'] = restored['ridge']
    for arm, index in selected.items():
        if index == 0 and not np.array_equal(result[arm], result['ridge']):
            raise ValueError('An off arm must reproduce original ridge probabilities')
    return result


def evaluate_fold(bank, scope, graphs, output, seed, fold, verify, smoke=False):
    config = scope['config']
    train = np.flatnonzero(bank.assigned != fold)
    splits = benchmark.inner_patient_splits(
        bank.labels[train], bank.ids[train], config['inner_splits'], config['split_seed'] + 1000 * fold)
    inner_contexts = []
    for inner, (ti, vi) in enumerate(splits, 1):
        inner_contexts.append(prepare_context(bank, scope, seed, fold, inner, train[ti], train[vi],
                                              range(len(config['ridge_alphas']))))
    baseline_scores = np.asarray([ctx.base_scores for ctx in inner_contexts]).T
    candidate, _ = core.select(baseline_scores)
    historical = artifacts.read(bank.root / 'review' / f'selection_{fold:02d}_baseline_s{seed}.json')
    if historical['selected_candidate'] != candidate or historical['inner_scores'] != baseline_scores.tolist():
        raise ValueError('Original ridge alpha choice differs')
    trials = [graph_context(bank, scope, output, ctx, graphs, candidate, verify) for ctx in inner_contexts]
    selected = {}
    for arm in graphs:
        selected[arm], _ = core.select(np.asarray([trial[arm] for trial in trials]).T)
    if smoke:
        return None, None, selected
    query = np.flatnonzero(bank.assigned == fold)
    outer = prepare_context(bank, scope, seed, fold, 0, train, query, [candidate])
    return query, graph_context(bank, scope, output, outer, graphs, candidate, verify, selected), selected


def make_scope(config, bank, graphs):
    from . import residual_context_report
    paths = [Path(module.__file__) for module in (
        benchmark, source, kernel, artifacts, core, residual_context_report)]
    paths.extend((Path(__file__), Path(__file__).with_name('subtype_graph_report.py'),
                  Path('scripts/run_subtype_graph.py')))
    scope = dict(config=config, source=bank.provenance, runtime=bank.source_scope['runtime'],
                 graph_laplacians={name: m.tolist() for name, m in graphs.items()},
                 code={str(path): file_hash(path) for path in paths})
    return dict(scope, study_fingerprint=_hash(scope))


def run(config, smoke=False, verify=False):
    graphs = make_graphs(config)
    bank, _ = source.validate_chain(config)
    root = Path(config['output_dir']) / ('smoke' if smoke else 'complete')
    output = root / 'review'
    output.mkdir(parents=True, exist_ok=True)
    scope = make_scope(config, bank, graphs)
    manifest = root / 'study_manifest.json'
    if manifest.exists() or verify:
        if artifacts.read(manifest) != scope:
            raise ValueError('Study scope changed. Use a new output namespace.')
    else:
        _save_json(manifest, scope)
    if smoke:
        evaluate_fold(bank, scope, graphs, output, config['projection_seeds'][0], 1, verify, True)
        if verify:
            _save_json(root / 'summary.json', dict(
                complete=True, outer_test_patients_used=False, feature_contexts_replayed=3,
                baseline_models_replayed=3 * len(config['ridge_alphas']),
                positive_graph_fits_replayed=3 * len(graphs) * (len(config['strengths']) - 1),
                study_fingerprint=scope['study_fingerprint']))
        print(f"Inner-only graph smoke {'replayed' if verify else 'fitted'}", flush=True)
        return
    members = {arm: {s: np.full((len(bank.ids), len(config['classes'])), np.nan)
                     for s in config['projection_seeds']} for arm in ARMS}
    counts = {arm: [0] * len(config['strengths']) for arm in graphs}
    for fold in range(1, config['outer_splits'] + 1):
        for seed in config['projection_seeds']:
            rows, values, selected = evaluate_fold(bank, scope, graphs, output, seed, fold, verify)
            for arm in graphs:
                counts[arm][selected[arm]] += 1
            for arm in ARMS:
                if np.isfinite(members[arm][seed][rows]).any():
                    raise ValueError('Duplicate outer patient predictions')
                members[arm][seed][rows] = values[arm]
        print(f"{'Replayed' if verify else 'Fitted'} fold {fold}/{config['outer_splits']}", flush=True)
    if verify:
        summarize(root, scope, bank, members, counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/subtype_graph_v1.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(config['torch_threads'])
    with threadpool_limits(limits=1):
        run(config, args.smoke, args.verify)


if __name__ == '__main__':
    main()
