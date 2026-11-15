"""Train-patient bootstrap consensus with fixed semantic feature groups."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import tensor_feature_bank as bank, tensor_feature_screen as original
from .benchmark import patient_balanced_weights
from .multiview_bank import array_hash
from .neural import _hash


def semantic_groups(names,family):
    keys = []
    for name in names:
        parts = name.split("/")
        kind = parts.pop(0) if family=="combined" else family
        # Keep summary / XYZ / band / window / statistic or coefficient order.
        # Only replace the physical view slot, never the median summary slot.
        if len(parts)>1 and parts[1] in bank.VIEWS:
            parts[1] = "corresponding_views"
        keys.append("/".join([kind]+parts))
    unique = list(dict.fromkeys(keys))
    lookup = {name:i for i,name in enumerate(unique)}
    return np.array([lookup[name] for name in keys],dtype=np.int64),unique


def bootstrap_draws(y,count,seed,panel):
    rng = np.random.default_rng(np.random.SeedSequence([seed,panel]))
    pools = [np.flatnonzero(y==label) for label in np.unique(y)]
    if len(pools)<2 or min(map(len,pools))<2:
        raise ValueError("Need at least two training patients in every class")
    return np.concatenate([rng.choice(pool,size=(count,len(pool)),replace=True) for pool in pools],axis=1)


def build_cache(X,y,weights,config,source,seed,draws,audit_draws):
    budget = max(source["feature_counts"])
    result = {}
    for key,panel in (("training",draws),("audit",audit_draws)):
        selected = np.full((len(panel),budget),-1,dtype=np.int64)
        for i,rows in enumerate(panel):
            # Repeated draws keep their multiplicity. Inverse duplicate-patient
            # weighting would cancel the bootstrap and is intentionally absent.
            selector = original.TrainingSelector().fit(X[rows],y[rows],weights[rows],budget,source["correlation_threshold"])
            selected[i,:len(selector.selected)] = selector.selected
        result[key] = selected
    return result


def bootstrap_cache(X,y,ids,train,names,family,config,source,seed,root,memo,verify=False):
    weights = patient_balanced_weights(y[train],ids[train])
    scope = {"training_features_sha256":array_hash(X[train]),"training_labels_sha256":array_hash(y[train]),
        "training_membership_sha256":_hash(ids[train].tolist()),"weights_sha256":array_hash(weights),
        "names_sha256":_hash(names),"family":family,"seed":seed,"bootstrap_replicates":config["bootstrap_replicates"],
        "audit_replicates":config["audit_replicates"],"maximum_budget":max(source["feature_counts"]),
        "correlation_threshold":source["correlation_threshold"],"selector_code_sha256":config["_selector_code_sha256"]}
    key = _hash(scope)
    if key in memo and (not verify or memo[key].get("verified")):
        return memo[key]
    root = Path(root)/"bootstrap_cache"
    root.mkdir(parents=True,exist_ok=True)
    path = root/f"{key}.npz"
    draws = bootstrap_draws(y[train],config["bootstrap_replicates"],seed,0)
    audit_draws = bootstrap_draws(y[train],config["audit_replicates"],seed,1)
    if path.exists():
        with np.load(path) as saved:
            if json.loads(str(saved["scope"].item()))!=scope or not np.array_equal(saved["draws"],draws) or not np.array_equal(saved["audit_draws"],audit_draws):
                raise ValueError("Bootstrap cache scope or training-only draws changed")
            selections = {name:saved[name] for name in ("training","audit")}
            if str(saved["digest"].item())!=_hash([array_hash(selections[name]) for name in ("training","audit")]):
                raise ValueError("Bootstrap cache selection content changed")
        if verify:
            expected = build_cache(X[train],y[train],weights,config,source,seed,draws,audit_draws)
            if any(not np.array_equal(expected[name],selections[name]) for name in expected):
                raise ValueError("Bootstrap cache does not reproduce resampled selectors")
    else:
        selections = build_cache(X[train],y[train],weights,config,source,seed,draws,audit_draws)
        temporary = path.with_suffix(".partial.npz")
        np.savez_compressed(temporary,scope=json.dumps(scope),draws=draws,audit_draws=audit_draws,
            digest=_hash([array_hash(selections[name]) for name in ("training","audit")]),**selections)
        temporary.replace(path)
    for name,count in (("training",config["bootstrap_replicates"]),("audit",config["audit_replicates"])):
        values = selections[name]
        if values.shape!=(count,max(source["feature_counts"])) or np.any(values < -1) or np.any(values>=X.shape[1]) or any(len(row[row>=0])!=len(np.unique(row[row>=0])) for row in values):
            raise ValueError("Invalid bootstrap selected-feature sequences")
    from .signal_audit import file_hash
    value = dict(selections,scope=scope,cache_sha256=file_hash(path),cache_key=key,verified=verify,
        draws_sha256=_hash([array_hash(draws),array_hash(audit_draws)]))
    memo[key] = value
    return value


def frequencies(rows,budget,p,group_ids):
    counts = np.zeros(p,dtype=np.int64)
    group_counts = np.zeros(group_ids.max()+1,dtype=np.int64)
    for row in rows:
        selected = row[:budget]
        selected = selected[selected>=0]
        counts[selected] += 1
        group_counts[np.unique(group_ids[selected])] += 1
    return counts/len(rows),group_counts/len(rows)


class ConsensusSelector:
    def fit(self,X,y,weights,budget,threshold,names,family,method,payload):
        baseline = original.TrainingSelector().fit(X,y,weights,budget,threshold)
        self.scale,self.scores = baseline.scale,baseline.scores
        self.method,self.budget = method,budget
        self.groups,self.group_names = semantic_groups(names,family)
        frequency,group_frequency = frequencies(payload["training"],budget,X.shape[1],self.groups)
        if method=="original":
            self.selected = baseline.selected
        else:
            scaled = self.scale.transform(X)
            variable = np.flatnonzero(np.ptp(scaled,axis=0)>1e-12)
            individual = sorted(variable,key=lambda i:(-frequency[i],-self.scores[i],int(i)))
            if method=="bootstrap":
                order = individual
            elif method=="group_bootstrap":
                members = {}
                for i in individual:
                    members.setdefault(int(self.groups[i]),[]).append(int(i))
                group_order = sorted(members,key=lambda g:(-group_frequency[g],-frequency[members[g][0]],-self.scores[members[g][0]],g))
                order = [i for group in group_order for i in members[group]]
            else:
                raise ValueError("Unknown consensus selector")
            centered = scaled-scaled.mean(0)
            length = np.linalg.norm(centered,axis=0)
            unit = np.divide(centered,length,out=np.zeros_like(centered),where=length>1e-12)
            selected,used_groups = [],set()
            for i in order:
                group = int(self.groups[i])
                if method=="group_bootstrap" and group in used_groups:
                    continue
                if selected and np.max(np.abs(unit[:,selected].T@unit[:,i]))>=threshold:
                    continue
                selected.append(int(i)); used_groups.add(group)
                if len(selected)>=budget:
                    break
            self.selected = np.array(selected,dtype=np.int64)
        self.base_audit = baseline.audit()
        self.frequency,self.group_frequency = frequency,group_frequency
        self.audit_frequency,self.audit_group_frequency = frequencies(payload["audit"],budget,X.shape[1],self.groups)
        self.cache_key,self.cache_sha256 = payload["cache_key"],payload["cache_sha256"]
        self.draws_sha256 = payload["draws_sha256"]
        self.bootstrap_replicates = len(payload["training"])
        self.audit_replicates = len(payload["audit"])
        return self

    def transform(self,X):
        return self.scale.transform(X)[:,self.selected]

    def audit(self):
        selected = self.selected
        return {"method":self.method,"selected_indices":selected.tolist(),"n_selected":len(selected),
            "scaler_mean_sha256":array_hash(self.scale.mean_),"scaler_scale_sha256":array_hash(self.scale.scale_),
            "training_ranking_sha256":array_hash(self.scores),"semantic_groups_sha256":_hash(self.group_names),
            "selected_group_ids":self.groups[selected].tolist(),"n_selected_groups":len(np.unique(self.groups[selected])),
            "training_selection_frequencies":self.frequency[selected].tolist(),
            "training_group_frequencies":self.group_frequency[self.groups[selected]].tolist(),
            "audit_selection_frequencies":self.audit_frequency[selected].tolist(),
            "audit_group_frequencies":self.audit_group_frequency[self.groups[selected]].tolist(),
            "bootstrap_cache_key":self.cache_key,"bootstrap_cache_sha256":self.cache_sha256,
            "bootstrap_replicates":self.bootstrap_replicates,"audit_replicates":self.audit_replicates,
            "sampling_scope_sha256":self.draws_sha256}
