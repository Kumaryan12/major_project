"""Compact beat-encoder/GRU models with record-bounded matched permutations."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from .benchmark import patient_balanced_weights
from .contrastive import initial_state_hash
from .neural import FoldNormalizer, VCGResNet, _device, _hash, _seed

ARMS = ("mean_rr", "ordered_no_rr", "ordered_rr", "shuffled_rr")


class SequenceNormalizer:
    def fit(self, bank, patients):
        rows = bank.beats_for(patients)
        weights = patient_balanced_weights(bank.y[rows], bank.groups[rows])
        self.wave = FoldNormalizer().fit(bank.waves, np.empty((len(bank.waves), 0), np.float32), rows, weights)
        valid = rows[bank.rr_valid[rows]]
        if len(valid):
            weights = patient_balanced_weights(bank.y[valid], bank.groups[valid])
            weights = weights/weights.sum()
            mean = float(weights@bank.rr[valid].astype(np.float64))
            variance = float(weights@np.square(bank.rr[valid].astype(np.float64)-mean))
            self.rr_mean, self.rr_scale = mean, float(np.sqrt(max(variance, 1e-12)))
        else:
            self.rr_mean, self.rr_scale = 0., 1.
        self.training_patients, self.fit_beats, self.valid_rr = len(patients), len(rows), len(valid)
        return self

    def audit(self):
        return {"wave": self.wave.audit(), "rr_mean_seconds": self.rr_mean, "rr_scale_seconds": self.rr_scale,
                "training_patients": self.training_patients, "fit_beats": self.fit_beats, "valid_rr": self.valid_rr}

    @classmethod
    def restore(cls, audit):
        result = cls();result.wave = FoldNormalizer()
        for key, value in audit["wave"].items():
            setattr(result.wave, key, np.asarray(value, np.float32))
        result.rr_mean, result.rr_scale = audit["rr_mean_seconds"], audit["rr_scale_seconds"]
        result.training_patients, result.fit_beats, result.valid_rr = audit["training_patients"], audit["fit_beats"], audit["valid_rr"]
        return result


def training_schedule(bank, patients, length, count, seed, epoch):
    """Class/patient balanced, uniform recording then uniform contiguous start."""
    patients = np.sort(np.asarray(patients))
    records = bank.records_for(patients)
    labels = np.array([bank.y[bank.rows(records[np.flatnonzero(bank.record_patients[records] == p)[0]])[0]] for p in patients])
    weights = patient_balanced_weights(labels, patients);weights /= weights.sum()
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0]))
    choices = rng.choice(len(patients), size=count, replace=True, p=weights)
    plan = []
    for choice in choices:
        candidates = records[bank.record_patients[records] == patients[choice]]
        record = int(rng.choice(candidates))
        size = len(bank.rows(record))
        start = int(rng.integers(max(size-length, 0)+1))
        plan.append((record, start, min(length, size)))
    return plan


def inference_schedule(bank, patients, length):
    plan, covered = [], []
    for record in bank.records_for(patients):
        rows = bank.rows(record)
        for start in range(0, len(rows), length):
            size = min(length, len(rows)-start)
            plan.append((int(record), start, size));covered.extend(rows[start:start+size].tolist())
    expected = bank.beats_for(patients)
    if not np.array_equal(np.sort(covered), expected):
        raise ValueError("Chronological inference does not cover every query beat exactly once")
    return plan, {"query_patients": len(patients), "query_records": len(bank.records_for(patients)),
                  "query_beats": len(expected), "windows": len(plan), "coverage_sha256": _hash(covered),
                  "plan_sha256": _hash(plan), "all_beats_exactly_once": True}


def permutation(size, seed, epoch, index):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, index, 77])).permutation(size)


class WindowDataset(Dataset):
    def __init__(self, bank, normalizer, plan, length, arm, seed, epoch):
        self.bank, self.normalizer, self.plan, self.length = bank, normalizer, plan, length
        self.arm, self.seed, self.epoch = arm, seed, epoch
        if arm not in ARMS or length < 1:
            raise ValueError("Invalid sequence model/length")

    def __len__(self):
        return len(self.plan)

    def order(self, index):
        record, start, size = self.plan[index]
        if self.arm != "shuffled_rr":
            return np.arange(size)
        key = int(_hash([record, start])[:16], 16) if self.epoch == 0 else index
        return permutation(size, self.seed, self.epoch, key)

    def __getitem__(self, index):
        record, start, size = self.plan[index]
        rows = self.bank.rows(record)[start:start+size]
        if len(rows) != size or size < 1 or size > self.length:
            raise ValueError("Window crosses its source recording")
        if self.arm == "shuffled_rr":
            rows = rows[self.order(index)]
        wave = np.zeros((self.length, 3, self.bank.waves.shape[-1]), np.float32)
        wave[:size] = (np.asarray(self.bank.waves[rows], np.float32)-self.normalizer.wave.wave_mean_[:, None])/self.normalizer.wave.wave_scale_[:, None]
        rr = np.zeros((self.length, 2), np.float32)
        valid = self.bank.rr_valid[rows]
        rr[:size, 0] = np.where(valid, (self.bank.rr[rows]-self.normalizer.rr_mean)/self.normalizer.rr_scale, 0)
        rr[:size, 1] = valid
        mask = np.arange(self.length) < size
        return torch.from_numpy(wave), torch.from_numpy(rr), torch.from_numpy(mask), int(self.bank.y[rows[0]])

    def audit(self):
        orders = [self.order(i).tolist() for i in range(len(self.plan))]
        return {"window_plan_sha256": _hash(self.plan), "token_order_sha256": _hash(orders),
                "windows": len(self.plan), "valid_beat_presentations": sum(size for _, _, size in self.plan),
                "nominal_beat_slots": len(self.plan)*self.length,
                "nonidentity_window_orders": sum(order != list(range(len(order))) for order in orders)}


class SequenceModel(nn.Module):
    def __init__(self, config, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Unknown chronological arm")
        base = VCGResNet(len(config["classes"]), tuple(config["widths"]), config["pool_bins"], config["dropout"])
        self.encoder, self.arm = base.encoder, arm
        width, hidden = 64+config["rr_embedding"], config["hidden_size"]
        self.rr_encoder = nn.Sequential(nn.Linear(2, config["rr_embedding"]), nn.GELU())
        self.hidden = hidden
        if arm == "mean_rr":
            self.projection = nn.Sequential(nn.Linear(width, hidden), nn.GELU())
        else:
            # Explicit standard GRU equations avoid platform-specific packed RNN kernels.
            self.input_gates = nn.Linear(width, 3*hidden)
            self.hidden_gates = nn.Linear(hidden, 3*hidden)
        self.head = nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(hidden, len(config["classes"])))

    def forward(self, waves, rr, mask):
        if waves.ndim != 4 or waves.shape[2] != 3 or rr.shape != (*waves.shape[:2], 2) or mask.shape != waves.shape[:2] or not mask.any(1).all():
            raise ValueError("Invalid masked chronological batch")
        batch, length = waves.shape[:2]
        embedding = self.encoder(waves.reshape(batch*length, 3, waves.shape[-1])).reshape(batch, length, 64)
        if self.arm == "ordered_no_rr":
            rr = torch.zeros_like(rr)
        tokens = torch.cat([embedding, self.rr_encoder(rr)], dim=-1)
        if self.arm == "mean_rr":
            z = (self.projection(tokens)*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)
        else:
            z = tokens.new_zeros((batch, self.hidden))
            for step in range(length):
                ir, iz, inn = self.input_gates(tokens[:, step]).chunk(3, dim=-1)
                hr, hz, hn = self.hidden_gates(z).chunk(3, dim=-1)
                reset, update = torch.sigmoid(ir+hr), torch.sigmoid(iz+hz)
                candidate = torch.tanh(inn+reset*hn)
                next_z = (1-update)*candidate+update*z
                z = torch.where(mask[:, step, None], next_z, z)
        return self.head(z)


def initial_model(config, seed, arm):
    _seed(seed, config)
    return SequenceModel(config, arm)


def predict(model, normalizer, bank, patients, config, seed):
    patients = np.asarray(patients)
    plan, audit = inference_schedule(bank, patients, config["sequence_length"])
    # No absolute times or indices enter the model. The fixed permutation key
    # depends on query-window identity, not the order of unrelated query patients.
    dataset = WindowDataset(bank, normalizer, plan, config["sequence_length"], model.arm, seed, 0)
    loader = DataLoader(dataset, batch_size=config["inference_batch_size"], shuffle=False, num_workers=0)
    values = []
    model.eval()
    with torch.no_grad():
        for x, rr, mask, _ in loader:
            p = model(x.to(_device(config)), rr.to(_device(config)), mask.to(_device(config))).softmax(1)
            values.append(p.cpu().numpy())
    values = np.concatenate(values)
    record_scores = {}
    for index, (record, _, size) in enumerate(plan):
        if record not in record_scores:
            record_scores[record] = np.zeros(len(config["classes"]), dtype=np.float64)
        record_scores[record] += values[index]*size/len(bank.rows(record))
    p = np.stack([np.mean([value for record, value in record_scores.items() if bank.record_patients[record] == patient], axis=0) for patient in patients])
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid chronological patient probabilities")
    audit["token_order_sha256"] = dataset.audit()["token_order_sha256"]
    return p, audit


def fit(bank, train, valid, config, seed, arm, epochs, progress=None):
    train = np.asarray(train)
    if valid is not None and set(train) & set(valid):
        raise ValueError("Chronology training/validation patient overlap")
    normalizer = SequenceNormalizer().fit(bank, train)
    model = initial_model(config, seed, arm).to(_device(config))
    initial = initial_state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best, best_epoch, state, best_p, best_coverage = -np.inf, 0, None, None, None
    history = []
    for epoch in range(1, epochs+1):
        plan = training_schedule(bank, train, config["sequence_length"], config["windows_per_epoch"], seed, epoch)
        dataset = WindowDataset(bank, normalizer, plan, config["sequence_length"], arm, seed, epoch)
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
        model.train();losses = []
        for x, rr, mask, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(_device(config)), rr.to(_device(config)), mask.to(_device(config)))
            loss = nn.functional.cross_entropy(logits, labels.to(_device(config)))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite chronological training loss")
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True);optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = dict(epoch=epoch, training_loss=float(np.mean(losses)), **dataset.audit())
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
        if progress:
            progress(row)
        if valid is not None and epoch-best_epoch >= config["patience"]:
            break
    if state is not None:
        model.load_state_dict(state)
    audit = {"seed": seed, "arm": arm, "initial_state_sha256": initial,
             "parameter_count": sum(p.numel() for p in model.parameters()), "normalization": normalizer.audit(),
             "selected_epoch": best_epoch if valid is not None else epochs,
             "best_validation_macro_f1": best if valid is not None else None, "history": history,
             "patient_overlap": 0, "train_patients": len(train), "validation_patients": len(valid) if valid is not None else 0}
    return model, normalizer, audit, best_p, best_coverage
