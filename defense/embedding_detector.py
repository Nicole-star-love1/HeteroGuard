# -*- coding: utf-8 -*-
"""
Embedding and gradient anomaly detectors.

These detectors use a reference model trained on aligned clean metadata, then
compare clean vs poisoned embeddings/logits.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from .utils import align_clean_to_poison_metadata, normalize_scores


class EmbeddingAnomalyDetector:
    def __init__(
        self,
        reference_model,
        clean_data: HeteroData,
        poison_data: HeteroData,
        target_type: str,
        device: str = "cpu",
    ):
        self.reference_model = reference_model
        self.clean_data = align_clean_to_poison_metadata(clean_data, poison_data).to(device)
        self.poison_data = poison_data.to(device)
        self.target_type = target_type
        self.device = device

    def score_embedding_deviation(self, train_idx: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        train_idx = train_idx.detach().cpu().long()
        model = self.reference_model
        model.eval()

        with torch.no_grad():
            emb_clean = model(self.clean_data, return_node_embeddings=True)[self.target_type]
            emb_poison = model(self.poison_data, return_node_embeddings=True)[self.target_type]

        n = min(emb_clean.size(0), emb_poison.size(0))
        valid = train_idx[train_idx < n]

        scores = torch.zeros(train_idx.numel(), dtype=torch.float)
        if valid.numel() == 0:
            return scores, {"embedding_valid": 0}

        diff = (emb_poison[:n] - emb_clean[:n]).norm(p=2, dim=1).detach().cpu()

        pos = {int(node): i for i, node in enumerate(train_idx.tolist())}
        for node in valid.tolist():
            scores[pos[int(node)]] = diff[int(node)]

        raw = scores.clone()
        return normalize_scores(scores), {
            "embedding_valid": int(valid.numel()),
            "raw_scores": raw,
        }

    def score_target_probability_shift(
        self,
        train_idx: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        train_idx = train_idx.detach().cpu().long()
        model = self.reference_model
        model.eval()

        with torch.no_grad():
            logits_clean = model(self.clean_data)[self.target_type]
            logits_poison = model(self.poison_data)[self.target_type]

        n = min(logits_clean.size(0), logits_poison.size(0))
        valid = train_idx[train_idx < n]

        scores = torch.zeros(train_idx.numel(), dtype=torch.float)
        if valid.numel() == 0:
            return scores, {"prob_shift_valid": 0}

        p_clean = F.softmax(logits_clean[:n], dim=1)
        p_poison = F.softmax(logits_poison[:n], dim=1)

        if target_class is None:
            shift = (p_poison - p_clean).clamp(min=0).max(dim=1).values
        else:
            shift = (p_poison[:, int(target_class)] - p_clean[:, int(target_class)]).clamp(min=0)

        shift = shift.detach().cpu()
        pos = {int(node): i for i, node in enumerate(train_idx.tolist())}
        for node in valid.tolist():
            scores[pos[int(node)]] = shift[int(node)]

        raw = scores.clone()
        return normalize_scores(scores), {
            "prob_shift_valid": int(valid.numel()),
            "raw_scores": raw,
        }


class GradientAnomalyDetector:
    def __init__(
        self,
        reference_model,
        poison_data: HeteroData,
        target_type: str,
        device: str = "cpu",
    ):
        self.reference_model = reference_model
        self.poison_data = poison_data.to(device)
        self.target_type = target_type
        self.device = device

    def score_gradient_norm(
        self,
        train_idx: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Computes gradient norm with respect to target-node embeddings.

        This is a proxy gradient signal. It does not require modifying raw
        HeteroData.x in-place.
        """
        train_idx = train_idx.detach().cpu().long()
        model = self.reference_model
        model.eval()

        with torch.no_grad():
            emb_dict = model(self.poison_data, return_node_embeddings=True)

        emb_target = emb_dict[self.target_type].detach().clone().requires_grad_(True)

        # Use classifier if the model exposes per-node-type classifiers.
        if not hasattr(model, "classifiers"):
            return torch.zeros(train_idx.numel()), {"gradient_available": False}

        classifier = model.classifiers[self.target_type]
        logits = classifier(emb_target)

        y = self.poison_data[self.target_type].y
        valid = train_idx[train_idx < logits.size(0)]
        if valid.numel() == 0:
            return torch.zeros(train_idx.numel()), {"gradient_available": True, "gradient_valid": 0}

        if target_class is None:
            loss = F.cross_entropy(logits[valid.to(logits.device)], y[valid.to(y.device)])
        else:
            log_prob = F.log_softmax(logits, dim=1)
            loss = log_prob[valid.to(logits.device), int(target_class)].sum()

        loss.backward()

        grad = emb_target.grad
        scores = torch.zeros(train_idx.numel(), dtype=torch.float)
        if grad is None:
            return scores, {"gradient_available": False}

        grad_norm = grad.norm(p=2, dim=1).detach().cpu()
        pos = {int(node): i for i, node in enumerate(train_idx.tolist())}
        for node in valid.tolist():
            scores[pos[int(node)]] = grad_norm[int(node)]

        raw = scores.clone()
        return normalize_scores(scores), {
            "gradient_available": True,
            "gradient_valid": int(valid.numel()),
            "raw_scores": raw,
        }
