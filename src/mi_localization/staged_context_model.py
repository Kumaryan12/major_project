"""Train context after partition-local morphology pretraining, with freeze controls."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from . import residual_context_model as base, chronology_model as chronology
from .contrastive import initial_state_hash
from .neural import _device

ARMS=("morphology","continued_morphology","joint_context","frozen_context","frozen_shuffled")
ADAPTER_ARMS=ARMS[1:]


def mapped_arm(arm):
    if arm=="continued_morphology":return "beat_only"
    if arm=="frozen_shuffled":return "shuffled_context"
    if arm in ("joint_context","frozen_context"):return "ordered_context"
    raise ValueError("Unknown staged adaptation arm")


def morphology_state(state):
    return {k:v for k,v in state.items() if k.startswith(("encoder.","beat_head."))}


def morphology_hash(model):
    # Same canonical tensor hash as the other neural integrity audits.
    import hashlib
    digest=hashlib.sha256()
    for name,value in sorted(morphology_state(model.state_dict()).items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize(config,seed,arm,teacher_state):
    model=base.initial_model(config,seed,mapped_arm(arm))
    expected=morphology_state(model.state_dict());teacher=morphology_state(teacher_state)
    if expected.keys()!=teacher.keys() or any(expected[k].shape!=teacher[k].shape for k in expected):
        raise ValueError("Incompatible partition-local morphology checkpoint")
    model.load_state_dict(dict(model.state_dict(),**teacher))
    # Exact zero correction initially: the initial predictor equals the teacher.
    nn.init.zeros_(model.context_head[1].weight);nn.init.zeros_(model.context_head[1].bias)
    for name,parameter in model.named_parameters():
        is_base=name.startswith(("encoder.","beat_head."))
        parameter.requires_grad_(True if arm=="joint_context" else is_base if arm=="continued_morphology" else not is_base)
    return model


def train_mode(model):
    model.train()
    # All stage-2 controls use the same deterministic direct path. Eval mode
    # disables direct-head dropout; it does not block joint-context gradients.
    model.encoder.eval();model.beat_head.eval()


def fit_adapter(bank,train,valid,config,seed,arm,epochs,teacher_state,normalization,progress=None):
    train=np.asarray(train)
    if arm not in ADAPTER_ARMS or epochs<0 or (valid is not None and set(train)&set(valid)):
        raise ValueError("Invalid staged partition/arm/epoch budget")
    norm=chronology.SequenceNormalizer.restore(normalization)
    if norm.training_patients!=len(train) or norm.fit_beats!=len(bank.beats_for(train)):
        raise ValueError("Staged teacher normalization membership changed")
    model=initialize(config,seed,arm,teacher_state).to(_device(config))
    initial=initial_state_hash(model);base_hash=morphology_hash(model)
    count=sum(p.numel() for p in model.parameters());active=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config["learning_rate"],weight_decay=config["weight_decay"])
    best,best_epoch,state,best_p,best_coverage=-np.inf,0,None,None,None
    initial_validation=None
    if valid is not None:
        p,coverage=base.predict(model,norm,bank,valid,config,seed)
        truth=bank.labels[np.searchsorted(bank.ids,valid)]
        best=initial_validation=float(f1_score(truth,p.argmax(1),labels=np.arange(len(config["classes"])),average="macro",zero_division=0))
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_p,best_coverage=p.copy(),coverage
    history=[]
    for epoch in range(1,epochs+1):
        plan=chronology.training_schedule(bank,train,config["sequence_length"],config["windows_per_epoch"],seed,epoch)
        dataset=base.ResidualDataset(bank,norm,plan,config["sequence_length"],model.arm,seed,epoch)
        loader=DataLoader(dataset,batch_size=config["batch_size"],shuffle=False,num_workers=0)
        train_mode(model);losses=[];gates=[]
        for x,rr,mask,labels,order in loader:
            x,rr,mask,labels,order=[v.to(_device(config)) for v in (x,rr,mask,labels,order)]
            optimizer.zero_grad(set_to_none=True)
            logp,_,gate=model(x,rr,mask,order);loss=F.nll_loss(logp,labels)
            if not torch.isfinite(loss):raise ValueError("Nonfinite staged loss")
            loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5,error_if_nonfinite=True);optimizer.step()
            losses.append(float(loss.detach().cpu()));gates.append(float(gate.detach().mean().cpu()))
        current=morphology_hash(model)
        if arm.startswith("frozen_") and current!=base_hash:raise ValueError("Frozen morphology weights changed during adaptation")
        row=dict(epoch=epoch,training_loss=float(np.mean(losses)),gate_mean=float(np.mean(gates)),morphology_sha256=current,**dataset.audit())
        if valid is not None:
            p,coverage=base.predict(model,norm,bank,valid,config,seed)
            truth=bank.labels[np.searchsorted(bank.ids,valid)]
            score=float(f1_score(truth,p.argmax(1),labels=np.arange(len(config["classes"])),average="macro",zero_division=0))
            row["validation_patient_macro_f1"]=score
            if score>best+1e-8:
                best,best_epoch=score,epoch
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_p,best_coverage=p.copy(),coverage
        history.append(row)
        if progress:progress(row)
        if valid is not None and epoch-best_epoch>=config["patience"]:break
    if state is not None:model.load_state_dict(state)
    audit=dict(phase="adaptation",arm=arm,seed=seed,initial_state_sha256=initial,initial_morphology_sha256=base_hash,
               final_morphology_sha256=morphology_hash(model),parameter_count=count,trainable_parameter_count=active,
               normalization=norm.audit(),initial_validation_macro_f1=initial_validation,
               best_validation_macro_f1=best if valid is not None else None,selected_epoch=best_epoch if valid is not None else epochs,
               history=history,patient_overlap=0,train_patients=len(train),validation_patients=len(valid) if valid is not None else 0,
               direct_dropout_enabled=False,fresh_optimizer=True,optimizer_steps=len(history)*(config["windows_per_epoch"]//config["batch_size"]))
    return model,norm,audit,best_p,best_coverage
