# -*- coding: utf-8 -*-
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData


def clone_hetero_data(data: HeteroData) -> HeteroData:
    out = HeteroData()
    for node_type in data.node_types:
        for key, value in data[node_type].items():
            out[node_type][key] = value.clone() if isinstance(value, torch.Tensor) else deepcopy(value)
        if "x" in out[node_type] and out[node_type].x is not None:
            out[node_type].num_nodes = int(out[node_type].x.size(0))
        elif getattr(data[node_type], "num_nodes", None) is not None:
            out[node_type].num_nodes = int(data[node_type].num_nodes)
    for edge_type in data.edge_types:
        for key, value in data[edge_type].items():
            out[edge_type][key] = value.clone() if isinstance(value, torch.Tensor) else deepcopy(value)
    return out


def purify_graph_by_triggers(poison_data: HeteroData, trigger_nodes_by_type: Optional[Dict[str, torch.Tensor]] = None) -> HeteroData:
    if not trigger_nodes_by_type:
        return clone_hetero_data(poison_data)

    data = clone_hetero_data(poison_data)
    trigger_sets = {}
    for node_type, nodes in trigger_nodes_by_type.items():
        if nodes is None:
            continue
        if not isinstance(nodes, torch.Tensor):
            nodes = torch.tensor(nodes, dtype=torch.long)
        trigger_sets[node_type] = set(nodes.detach().cpu().long().tolist())

    for edge_type in list(data.edge_types):
        src_type, _, dst_type = edge_type
        if "edge_index" not in data[edge_type]:
            continue
        edge_index = data[edge_type].edge_index
        if edge_index.numel() == 0:
            continue

        src_bad = trigger_sets.get(src_type, set())
        dst_bad = trigger_sets.get(dst_type, set())
        if not src_bad and not dst_bad:
            continue

        src_cpu = edge_index[0].detach().cpu()
        dst_cpu = edge_index[1].detach().cpu()
        keep = torch.ones(edge_index.size(1), dtype=torch.bool)

        if src_bad:
            src_mask = torch.tensor([int(v) in src_bad for v in src_cpu.tolist()], dtype=torch.bool)
            keep &= ~src_mask
        if dst_bad:
            dst_mask = torch.tensor([int(v) in dst_bad for v in dst_cpu.tolist()], dtype=torch.bool)
            keep &= ~dst_mask

        data[edge_type].edge_index = edge_index[:, keep.to(edge_index.device)].contiguous()
    return data


class DefenseTrainer:
    def __init__(self, model, data: HeteroData, target_type: str, device: str = "cpu"):
        self.model = model
        self.data = data
        self.target_type = target_type
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.data = self.data.to(self.device)
        self.model = self.model.to(self.device)

    def _lazy_build(self, extra_data: Optional[HeteroData] = None):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(self.data)
            if extra_data is not None:
                _ = self.model(extra_data.to(self.device))
        self.model.to(self.device)

    def _train_indices(self) -> torch.Tensor:
        return torch.where(self.data[self.target_type].train_mask)[0].long().to(self.device)

    def _val_accuracy(self) -> float:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data)[self.target_type]
            y = self.data[self.target_type].y
            val_mask = self.data[self.target_type].val_mask
            if val_mask.sum().item() == 0:
                return 0.0
            pred = logits[val_mask].argmax(dim=1)
            return float((pred == y[val_mask]).float().mean().item())

    def _make_weights(self, suspicious_scores, train_idx, ce_len: int, min_weight: float, max_downweight: float) -> torch.Tensor:
        if suspicious_scores is None:
            return torch.ones(ce_len, dtype=torch.float, device=self.device)

        scores = suspicious_scores.detach().float().to(self.device).view(-1)

        if scores.numel() == ce_len:
            score_train = scores
        elif scores.numel() == self.data[self.target_type].x.size(0):
            score_train = scores[train_idx]
        else:
            score_train = torch.zeros(ce_len, dtype=torch.float, device=self.device)

        score_train = torch.nan_to_num(score_train, nan=0.0, posinf=1.0, neginf=0.0)
        if score_train.numel() > 0 and (score_train.max() > 1.0 or score_train.min() < 0.0):
            denom = score_train.max() - score_train.min()
            score_train = (score_train - score_train.min()) / denom.clamp(min=1e-12)

        weights = 1.0 - float(max_downweight) * score_train
        return weights.clamp(min=float(min_weight), max=1.0)

    def _sample_unlearn_indices(self, train_idx, target_class, unlearn_samples: int, exclude_target: bool = True) -> torch.Tensor:
        if train_idx.numel() == 0:
            return train_idx
        y = self.data[self.target_type].y
        candidates = train_idx
        if exclude_target and target_class is not None:
            mask = y[candidates] != int(target_class)
            candidates = candidates[mask]
        if candidates.numel() == 0:
            candidates = train_idx
        n = min(int(unlearn_samples), int(candidates.numel()))
        if n <= 0:
            return candidates[:0]
        perm = torch.randperm(candidates.numel(), device=candidates.device)[:n]
        return candidates[perm].long()

    def fit_reference(self, epochs: int = 50, lr: float = 0.005, weight_decay: float = 1e-4, verbose: bool = False):
        self._lazy_build()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        train_idx = self._train_indices()
        y = self.data[self.target_type].y

        best_state = None
        best_val = -1.0
        best_epoch = 0

        for epoch in range(int(epochs)):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(self.data)[self.target_type]
            loss = F.cross_entropy(logits[train_idx], y[train_idx])
            loss.backward()
            optimizer.step()

            val_acc = self._val_accuracy()
            if val_acc > best_val:
                best_val = val_acc
                best_state = deepcopy(self.model.state_dict())
                best_epoch = epoch

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return {"best_epoch": int(best_epoch), "best_val_acc": float(best_val)}

    def fit_weighted(self, suspicious_scores, epochs: int = 100, lr: float = 0.005, weight_decay: float = 1e-4, min_weight: float = 0.1, max_downweight: float = 0.9, verbose: bool = False):
        self._lazy_build()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        train_idx = self._train_indices()
        y = self.data[self.target_type].y

        best_state = None
        best_val = -1.0
        best_epoch = 0
        last_loss = 0.0

        for epoch in range(int(epochs)):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(self.data)[self.target_type]
            ce = F.cross_entropy(logits[train_idx], y[train_idx], reduction="none")
            weights = self._make_weights(suspicious_scores, train_idx, ce.numel(), min_weight, max_downweight)
            loss = (ce * weights).sum() / weights.sum().clamp(min=1e-12)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

            val_acc = self._val_accuracy()
            if val_acc > best_val:
                best_val = val_acc
                best_state = deepcopy(self.model.state_dict())
                best_epoch = epoch

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return {"best_epoch": int(best_epoch), "best_val_acc": float(best_val), "last_loss": float(last_loss), "mode": "weighted"}

    def fit_weighted_unlearn(
        self,
        suspicious_scores,
        attacker,
        target_class: int,
        epochs: int = 100,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        min_weight: float = 0.1,
        max_downweight: float = 0.9,
        unlearn_lambda: float = 1.0,
        unlearn_samples: int = 256,
        target_suppression: float = 0.1,
        unlearn_exclude_target: bool = True,
        verbose: bool = False,
    ):
        train_idx = self._train_indices()
        y = self.data[self.target_type].y

        unlearn_idx = self._sample_unlearn_indices(
            train_idx=train_idx,
            target_class=target_class,
            unlearn_samples=unlearn_samples,
            exclude_target=unlearn_exclude_target,
        )

        triggered_data = None
        if attacker is not None and unlearn_idx.numel() > 0:
            triggered_data = attacker.inject_trigger(
                clone_hetero_data(self.data).cpu(),
                unlearn_idx.detach().cpu(),
            ).to(self.device)

        self._lazy_build(extra_data=triggered_data)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_state = None
        best_val = -1.0
        best_epoch = 0
        last_loss = 0.0
        last_clean_loss = 0.0
        last_unlearn_loss = 0.0
        last_suppress_loss = 0.0

        for epoch in range(int(epochs)):
            self.model.train()
            optimizer.zero_grad()

            logits = self.model(self.data)[self.target_type]
            ce = F.cross_entropy(logits[train_idx], y[train_idx], reduction="none")
            weights = self._make_weights(suspicious_scores, train_idx, ce.numel(), min_weight, max_downweight)
            clean_loss = (ce * weights).sum() / weights.sum().clamp(min=1e-12)

            unlearn_loss = torch.tensor(0.0, dtype=clean_loss.dtype, device=self.device)
            suppress_loss = torch.tensor(0.0, dtype=clean_loss.dtype, device=self.device)

            if triggered_data is not None and unlearn_idx.numel() > 0:
                trig_logits = self.model(triggered_data)[self.target_type]
                unlearn_loss = F.cross_entropy(trig_logits[unlearn_idx], y[unlearn_idx])
                if target_class is not None and 0 <= int(target_class) < trig_logits.size(1):
                    target_prob = F.softmax(trig_logits[unlearn_idx], dim=1)[:, int(target_class)]
                    suppress_loss = target_prob.mean()

            loss = clean_loss + float(unlearn_lambda) * unlearn_loss + float(target_suppression) * suppress_loss
            loss.backward()
            optimizer.step()

            last_loss = float(loss.item())
            last_clean_loss = float(clean_loss.item())
            last_unlearn_loss = float(unlearn_loss.item())
            last_suppress_loss = float(suppress_loss.item())

            val_acc = self._val_accuracy()
            if val_acc > best_val:
                best_val = val_acc
                best_state = deepcopy(self.model.state_dict())
                best_epoch = epoch

            if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                print(f"[DefenseTrainer:unlearn] epoch={epoch+1}, loss={last_loss:.4f}, clean={last_clean_loss:.4f}, unlearn={last_unlearn_loss:.4f}, suppress={last_suppress_loss:.4f}, val_acc={val_acc:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return {
            "best_epoch": int(best_epoch),
            "best_val_acc": float(best_val),
            "last_loss": float(last_loss),
            "last_clean_loss": float(last_clean_loss),
            "last_unlearn_loss": float(last_unlearn_loss),
            "last_suppress_loss": float(last_suppress_loss),
            "unlearn_samples": int(unlearn_idx.numel()),
            "unlearn_lambda": float(unlearn_lambda),
            "target_suppression": float(target_suppression),
            "mode": "weighted_unlearn",
        }
