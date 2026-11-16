"""Direct beat probabilities with an optional gated contextual logit residual."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from . import chronology_model as chronology
from .contrastive import initial_state_hash
from .neural import VCGResNet, _device, _hash, _seed

ARMS = ("beat_only", "beat_aux", "ordered_context", "ordered_context_aux",
        "shuffled_context", "shuffled_context_aux")


def auxiliary_weight(config, arm):
    if arm not in ARMS:
        raise ValueError("Unknown residual-context arm")
    return float(config["auxiliary_weight"]) if arm.endswith("_aux") else 0.


class ResidualDataset(chronology.WindowDataset):
    """Keep direct beats fixed; shuffle only embedding/RR pairs in context."""
    def __init__(self, bank, normalizer, plan, length, arm, seed, epoch):
        if arm not in ARMS:
            raise ValueError("Unknown residual-context dataset arm")
        super().__init__(bank, normalizer, plan, length, "ordered_rr", seed, epoch)
        self.residual_arm = arm

    def order(self, index):
        record, start, size = self.plan[index]
        if not self.residual_arm.startswith("shuffled_"):
            return np.arange(size)
        key = int(_hash([record, start])[:16], 16) if self.epoch == 0 else index
        return chronology.permutation(size, self.seed, self.epoch, key)

    def __getitem__(self, index):
        x, rr, mask, label = super().__getitem__(index)
        size = self.plan[index][2]
        order = np.arange(self.length, dtype=np.int64)
        order[:size] = self.order(index)
        return x, rr, mask, label, torch.from_numpy(order)

    def audit(self):
        value = super().audit()
        value["direct_token_order_sha256"] = _hash([list(range(size)) for _, _, size in self.plan])
        return value


class ResidualContextModel(nn.Module):
    def __init__(self, config, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Unknown residual-context model arm")
        base = VCGResNet(len(config["classes"]), tuple(config["widths"]), config["pool_bins"], config["dropout"])
        self.encoder, self.beat_head, self.arm = base.encoder, base.classifier, arm
        self.hidden = config["hidden_size"]
        width = 64+config["rr_embedding"]
        self.rr_encoder = nn.Sequential(nn.Linear(2, config["rr_embedding"]), nn.GELU())
        self.input_gates = nn.Linear(width, 3*self.hidden)
        self.hidden_gates = nn.Linear(self.hidden, 3*self.hidden)
        self.context_head = nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(self.hidden, len(config["classes"])))
        self.gate = nn.Linear(64+self.hidden, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, config["gate_initial_bias"])

    def forward(self, waves, rr, mask, order):
        if (waves.ndim != 4 or waves.shape[2] != 3 or rr.shape != (*waves.shape[:2], 2)
                or mask.shape != waves.shape[:2] or order.shape != mask.shape or not mask.any(1).all()):
            raise ValueError("Invalid residual-context masked batch")
        batch, length = mask.shape
        embedding = self.encoder(waves.reshape(batch*length, 3, waves.shape[-1])).reshape(batch, length, 64)
        beat_logits = self.beat_head(embedding)
        # Context sees the same waveform/RR pairs, permuted jointly. Direct
        # logits and dropout assignments never change under the control order.
        tokens = torch.cat([embedding, self.rr_encoder(rr)], dim=-1)
        tokens = tokens.gather(1, order[..., None].expand_as(tokens))
        context_mask = mask.gather(1, order)
        z = tokens.new_zeros((batch, self.hidden))
        for step in range(length):
            ir, iz, inn = self.input_gates(tokens[:, step]).chunk(3, dim=-1)
            hr, hz, hn = self.hidden_gates(z).chunk(3, dim=-1)
            reset, update = torch.sigmoid(ir+hr), torch.sigmoid(iz+hz)
            candidate = torch.tanh(inn+reset*hn)
            z = torch.where(context_mask[:, step, None], (1-update)*candidate+update*z, z)
        mean = (embedding*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)
        gate = torch.sigmoid(self.gate(torch.cat([mean, z], dim=-1)))
        if self.arm.startswith("beat_"):
            gate = gate*0
        correction = gate*self.context_head(z)
        corrected = beat_logits+correction[:, None, :]
        log_p = corrected.log_softmax(-1).masked_fill(~mask[..., None], -torch.inf)
        window_log_p = torch.logsumexp(log_p, dim=1)-mask.sum(1).log()[:, None]
        return window_log_p, beat_logits, gate


def objective(log_p, beat_logits, labels, mask, weight):
    window = F.nll_loss(log_p, labels)
    targets = labels[:, None].expand(mask.shape)
    losses = F.cross_entropy(beat_logits.reshape(-1, beat_logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(mask.shape)
    beat = ((losses*mask).sum(1)/mask.sum(1)).mean()
    return window+weight*beat, window, beat


def initial_model(config, seed, arm):
    _seed(seed, config)
    return ResidualContextModel(config, arm)


def predict(model, normalizer, bank, patients, config, seed):
    patients = np.asarray(patients)
    plan, audit = chronology.inference_schedule(bank, patients, config["sequence_length"])
    dataset = ResidualDataset(bank, normalizer, plan, config["sequence_length"], model.arm, seed, 0)
    loader = DataLoader(dataset, batch_size=config["inference_batch_size"], shuffle=False, num_workers=0)
    values = []
    device = _device(config)
    model.eval()
    with torch.no_grad():
        for x, rr, mask, _, order in loader:
            log_p, _, _ = model(x.to(device), rr.to(device), mask.to(device), order.to(device))
            values.append(log_p.exp().cpu().numpy())
    values = np.concatenate(values)
    record_scores = {}
    for index, (record, _, size) in enumerate(plan):
        record_scores.setdefault(record, np.zeros(len(config["classes"]), dtype=np.float64))
        record_scores[record] += values[index]*size/len(bank.rows(record))
    p = np.stack([np.mean([value for record, value in record_scores.items() if bank.record_patients[record] == patient], axis=0) for patient in patients])
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid residual-context patient probabilities")
    audit.update({k: v for k, v in dataset.audit().items() if "order_sha256" in k})
    return p, audit


def fit(bank, train, valid, config, seed, arm, epochs, progress=None):
    train = np.asarray(train)
    if valid is not None and set(train) & set(valid):
        raise ValueError("Residual-context training/validation patient overlap")
    normalizer = chronology.SequenceNormalizer().fit(bank, train)
    model = initial_model(config, seed, arm).to(_device(config))
    initial = initial_state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    weight = auxiliary_weight(config, arm)
    best, best_epoch, state, best_p, best_coverage = -np.inf, 0, None, None, None
    history = []
    for epoch in range(1, epochs+1):
        plan = chronology.training_schedule(bank, train, config["sequence_length"], config["windows_per_epoch"], seed, epoch)
        dataset = ResidualDataset(bank, normalizer, plan, config["sequence_length"], arm, seed, epoch)
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
        model.train();losses, window_losses, beat_losses, gates = [], [], [], []
        device = _device(config)
        for x, rr, mask, labels, order in loader:
            x, rr, mask, labels, order = [v.to(device) for v in (x, rr, mask, labels, order)]
            optimizer.zero_grad(set_to_none=True)
            log_p, beat_logits, gate = model(x, rr, mask, order)
            loss, window, beat = objective(log_p, beat_logits, labels, mask, weight)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite residual-context loss")
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True);optimizer.step()
            losses.append(float(loss.detach().cpu()));window_losses.append(float(window.detach().cpu()))
            beat_losses.append(float(beat.detach().cpu()));gates.append(float(gate.detach().mean().cpu()))
        row = dict(epoch=epoch, training_loss=float(np.mean(losses)), window_loss=float(np.mean(window_losses)),
                   beat_loss=float(np.mean(beat_losses)), gate_mean=float(np.mean(gates)), **dataset.audit())
        if valid is not None:
            p, coverage = predict(model, normalizer, bank, valid, config, seed)
            truth = bank.labels[np.searchsorted(bank.ids, valid)]
            score = float(f1_score(truth, p.argmax(1), labels=np.arange(len(config["classes"])), average="macro", zero_division=0))
            row["validation_patient_macro_f1"] = score
            if score > best+1e-8:
                best, best_epoch = score, epoch
                state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_p, best_coverage = p.copy(), coverage
        history.append(row)
        if progress: progress(row)
        if valid is not None and epoch-best_epoch >= config["patience"]: break
    if state is not None: model.load_state_dict(state)
    audit = {"seed": seed, "arm": arm, "initial_state_sha256": initial, "parameter_count": sum(p.numel() for p in model.parameters()),
             "normalization": normalizer.audit(), "auxiliary_weight": weight, "gate_initial_bias": config["gate_initial_bias"],
             "selected_epoch": best_epoch if valid is not None else epochs, "best_validation_macro_f1": best if valid is not None else None,
             "history": history, "patient_overlap": 0, "train_patients": len(train), "validation_patients": len(valid) if valid is not None else 0}
    return model, normalizer, audit, best_p, best_coverage
