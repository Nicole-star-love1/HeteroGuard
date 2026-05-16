# -*- coding: utf-8 -*-
"""
Unified multi-signal poison detector.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch_geometric.data import HeteroData

from .embedding_detector import EmbeddingAnomalyDetector, GradientAnomalyDetector
from .feature_detector import FeatureAnomalyDetector
from .structural_detector import RelationAwareStructuralDetector
from .utils import compute_binary_metrics, get_target_train_idx, normalize_scores, summarize_scores


class HeteroPoisonDetector:
    """
    Multi-signal detector for suspicious target-type training nodes.

    Signals:
        - structural: new trigger-node/edge adjacency
        - feature: clean-vs-poison target feature difference
        - embedding: clean-vs-poison embedding deviation
        - prob_shift: target-class probability shift
        - gradient: gradient norm anomaly
        - label: optional oracle diagnostic signal
    """

    DEFAULT_WEIGHTS = {
        "structural": 4.0,
        "feature": 2.0,
        "embedding": 2.0,
        "prob_shift": 1.0,
        "gradient": 1.0,
        "label": 0.0,  # diagnostic only by default
    }

    def __init__(
        self,
        clean_data: HeteroData,
        poison_data: HeteroData,
        target_type: str,
        reference_model=None,
        device: str = "cpu",
    ):
        self.clean_data = clean_data
        self.poison_data = poison_data
        self.target_type = target_type
        self.reference_model = reference_model
        self.device = device

        self.last_signal_scores: Dict[str, torch.Tensor] = {}
        self.last_signal_info: Dict[str, Dict] = {}
        self.last_final_scores: Optional[torch.Tensor] = None
        self.last_train_idx: Optional[torch.Tensor] = None

    def score(
        self,
        target_class: Optional[int] = None,
        disable_signals: Optional[Sequence[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        use_label_signal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        disabled = set(disable_signals or [])
        w = dict(self.DEFAULT_WEIGHTS)
        if weights:
            w.update(weights)
        if use_label_signal and "label" not in disabled:
            w["label"] = max(w.get("label", 0.0), 3.0)

        train_idx = get_target_train_idx(self.poison_data, self.target_type)
        final = torch.zeros(train_idx.numel(), dtype=torch.float)
        signal_scores = {}
        signal_info = {}

        # 1. Relation-aware structural signal.
        if "structural" not in disabled:
            structural = RelationAwareStructuralDetector(
                clean_data=self.clean_data,
                poison_data=self.poison_data,
                target_type=self.target_type,
            )
            s, info = structural.score_target_nodes(train_idx)
            signal_scores["structural"] = s
            signal_info["structural"] = info
            final += float(w.get("structural", 0.0)) * s

        # 2. Feature anomaly signal.
        if "feature" not in disabled:
            feat = FeatureAnomalyDetector(
                clean_data=self.clean_data,
                poison_data=self.poison_data,
                target_type=self.target_type,
            )
            s, info = feat.score_target_nodes(train_idx)
            signal_scores["feature"] = s
            signal_info["feature"] = info
            final += float(w.get("feature", 0.0)) * s

        # 3. Optional label oracle diagnostic signal.
        if use_label_signal and "label" not in disabled:
            s, info = self._score_label_change(train_idx)
            signal_scores["label"] = s
            signal_info["label"] = info
            final += float(w.get("label", 0.0)) * s

        # 4. Embedding/prob/gradient signals require a reference model.
        if self.reference_model is not None:
            if "embedding" not in disabled or "prob_shift" not in disabled:
                emb = EmbeddingAnomalyDetector(
                    reference_model=self.reference_model,
                    clean_data=self.clean_data,
                    poison_data=self.poison_data,
                    target_type=self.target_type,
                    device=self.device,
                )

                if "embedding" not in disabled:
                    s, info = emb.score_embedding_deviation(train_idx)
                    signal_scores["embedding"] = s
                    signal_info["embedding"] = info
                    final += float(w.get("embedding", 0.0)) * s

                if "prob_shift" not in disabled:
                    s, info = emb.score_target_probability_shift(
                        train_idx=train_idx,
                        target_class=target_class,
                    )
                    signal_scores["prob_shift"] = s
                    signal_info["prob_shift"] = info
                    final += float(w.get("prob_shift", 0.0)) * s

            if "gradient" not in disabled:
                grad = GradientAnomalyDetector(
                    reference_model=self.reference_model,
                    poison_data=self.poison_data,
                    target_type=self.target_type,
                    device=self.device,
                )
                s, info = grad.score_gradient_norm(
                    train_idx=train_idx,
                    target_class=target_class,
                )
                signal_scores["gradient"] = s
                signal_info["gradient"] = info
                final += float(w.get("gradient", 0.0)) * s

        final = normalize_scores(final)

        self.last_signal_scores = signal_scores
        self.last_signal_info = signal_info
        self.last_final_scores = final
        self.last_train_idx = train_idx

        summary = {
            "signal_summary": {
                name: summarize_scores(name, scores)
                for name, scores in signal_scores.items()
            },
            "signal_info": signal_info,
            "weights": w,
            "disabled": list(disabled),
            "use_label_signal": bool(use_label_signal),
        }

        return train_idx, final, summary

    def select_suspicious(
        self,
        scores: torch.Tensor,
        train_idx: torch.Tensor,
        top_k_ratio: float = 0.1,
        min_k: int = 1,
    ) -> torch.Tensor:
        n = scores.numel()
        if n == 0:
            return torch.empty(0, dtype=torch.long)

        k = max(int(min_k), int(round(n * float(top_k_ratio))))
        k = min(k, n)

        top_pos = torch.argsort(scores, descending=True)[:k]
        return train_idx[top_pos].long()

    def detect(
        self,
        target_class: Optional[int] = None,
        top_k_ratio: float = 0.1,
        true_poison_indices: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
        **kwargs,
    ):
        train_idx, scores, info = self.score(target_class=target_class, **kwargs)
        suspicious = self.select_suspicious(scores, train_idx, top_k_ratio=top_k_ratio)

        if return_metrics:
            metrics = compute_binary_metrics(
                predicted_nodes=suspicious,
                true_nodes=true_poison_indices if true_poison_indices is not None else [],
            )
            info["metrics"] = metrics
            return suspicious, scores, metrics, info

        return suspicious, scores, info

    def _score_label_change(self, train_idx: torch.Tensor):
        clean_y = self.clean_data[self.target_type].y.detach().cpu()
        poison_y = self.poison_data[self.target_type].y.detach().cpu()
        n = min(clean_y.size(0), poison_y.size(0))

        scores = torch.zeros(train_idx.numel(), dtype=torch.float)
        pos = {int(node): i for i, node in enumerate(train_idx.tolist())}

        changed = 0
        for node in train_idx.tolist():
            node = int(node)
            if node < n and int(clean_y[node]) != int(poison_y[node]):
                scores[pos[node]] = 1.0
                changed += 1

        return scores, {"label_changed_train_nodes": int(changed)}
