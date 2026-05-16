# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch_geometric.data import HeteroData

from .detector import HeteroPoisonDetector
from .structural_detector import RelationAwareStructuralDetector
from .trainer import DefenseTrainer, purify_graph_by_triggers
from .utils import align_clean_to_poison_metadata


class HeteroGuard:
    def __init__(
        self,
        data: Optional[HeteroData] = None,
        target_node_type: Optional[str] = None,
        num_classes: Optional[int] = None,
        hidden_dim: int = 128,
        device: str = "cpu",
        train_data: Optional[HeteroData] = None,
        poison_data: Optional[HeteroData] = None,
        model_name: str = "HAN",
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_bases: int = 8,
        **kwargs,
    ):
        clean_data = kwargs.pop("clean_data", None)
        if clean_data is None:
            clean_data = train_data if train_data is not None else data
        if poison_data is None:
            poison_data = data
        if clean_data is None or poison_data is None:
            raise ValueError("Both clean_data/train_data and poison_data/data must be provided.")
        if target_node_type is None:
            raise ValueError("target_node_type must be provided.")
        if num_classes is None:
            raise ValueError("num_classes must be provided.")

        self.clean_data = clean_data
        self.poison_data = poison_data
        self.target_type = target_node_type
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.device = device
        self.model_name = model_name
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.num_bases = int(num_bases)

        self.reference_model = None
        self.defense_model = None
        self.suspicious_nodes = None
        self.suspicious_scores = None
        self.detect_info = {}
        self.trigger_nodes_by_type = {}
        self.purified_data = None
        self.hard_removed_train_nodes = 0

    def _create_model(self, data: HeteroData):
        from models.hetero_gnn import create_model
        return create_model(
            model_name=self.model_name,
            data=data,
            num_classes=self.num_classes,
            target_node_type=self.target_type,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_bases=self.num_bases,
        )

    def pretrain_reference(self, epochs: int = 50, lr: float = 0.005, weight_decay: float = 1e-4, verbose: bool = False):
        aligned_clean = align_clean_to_poison_metadata(self.clean_data, self.poison_data)
        model = self._create_model(aligned_clean)
        trainer = DefenseTrainer(model=model, data=aligned_clean, target_type=self.target_type, device=self.device)
        info = trainer.fit_reference(epochs=epochs, lr=lr, weight_decay=weight_decay, verbose=verbose)
        self.reference_model = trainer.model
        return info

    def pretrain_ssl(self, epochs: int = 50, verbose: bool = True, train_data: Optional[HeteroData] = None):
        if train_data is not None:
            self.clean_data = train_data
        return self.pretrain_reference(epochs=epochs, verbose=verbose)

    def detect(
        self,
        clean_data: Optional[HeteroData] = None,
        top_k_ratio: float = 0.1,
        target_class: Optional[int] = None,
        return_metrics: bool = False,
        true_poison_indices: Optional[torch.Tensor] = None,
        adaptive_threshold: bool = True,
        disable_signals: Optional[Sequence[str]] = None,
        use_label_signal: bool = False,
        weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        if clean_data is not None:
            self.clean_data = clean_data
        if self.reference_model is None:
            self.pretrain_reference(epochs=30, verbose=False)

        detector = HeteroPoisonDetector(
            clean_data=self.clean_data,
            poison_data=self.poison_data,
            target_type=self.target_type,
            reference_model=self.reference_model,
            device=self.device,
        )
        result = detector.detect(
            target_class=target_class,
            top_k_ratio=top_k_ratio,
            true_poison_indices=true_poison_indices,
            return_metrics=return_metrics,
            disable_signals=disable_signals,
            use_label_signal=use_label_signal,
            weights=weights,
        )
        if return_metrics:
            suspicious, scores, metrics, info = result
            self.detect_info = info
        else:
            suspicious, scores, info = result
            metrics = None
            self.detect_info = info
        self.suspicious_nodes = suspicious
        self.suspicious_scores = scores

        structural = RelationAwareStructuralDetector(clean_data=self.clean_data, poison_data=self.poison_data, target_type=self.target_type)
        self.trigger_nodes_by_type = structural.detect_trigger_nodes_by_type()
        if return_metrics:
            return suspicious, scores, metrics
        return suspicious, scores

    def purify(self):
        if not self.trigger_nodes_by_type:
            structural = RelationAwareStructuralDetector(clean_data=self.clean_data, poison_data=self.poison_data, target_type=self.target_type)
            self.trigger_nodes_by_type = structural.detect_trigger_nodes_by_type()
        self.purified_data = purify_graph_by_triggers(self.poison_data, self.trigger_nodes_by_type)
        return self.purified_data

    def _hard_remove_suspicious_from_train_mask(self, train_graph: HeteroData) -> int:
        if self.suspicious_nodes is None or len(self.suspicious_nodes) == 0:
            return 0
        if "train_mask" not in train_graph[self.target_type]:
            return 0
        mask = train_graph[self.target_type].train_mask.clone()
        suspicious = self.suspicious_nodes.detach().cpu().long()
        suspicious = suspicious[suspicious < mask.size(0)]
        if suspicious.numel() == 0:
            return 0
        before = int(mask.sum().item())
        mask[suspicious.to(mask.device)] = False
        after = int(mask.sum().item())
        train_graph[self.target_type].train_mask = mask
        return int(before - after)

    def train_defense(
        self,
        epochs: int = 100,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        verbose: bool = False,
        clean_data: Optional[HeteroData] = None,
        fresh_start: bool = True,
        use_clean_graph: bool = False,
        use_prune: bool = True,
        min_weight: float = 0.1,
        max_downweight: float = 0.9,
        hard_remove_suspicious: bool = True,
        use_trigger_unlearning: bool = False,
        attacker=None,
        target_class: Optional[int] = None,
        unlearn_lambda: float = 1.0,
        unlearn_samples: int = 256,
        target_suppression: float = 0.1,
        unlearn_exclude_target: bool = True,
        **kwargs,
    ):
        if clean_data is not None:
            self.clean_data = clean_data
        if self.suspicious_scores is None:
            self.detect(top_k_ratio=0.1, return_metrics=False)

        if use_clean_graph:
            train_graph = align_clean_to_poison_metadata(self.clean_data, self.poison_data)
        elif use_prune:
            train_graph = self.purify()
        else:
            train_graph = self.poison_data

        self.hard_removed_train_nodes = 0
        if hard_remove_suspicious and not use_clean_graph:
            self.hard_removed_train_nodes = self._hard_remove_suspicious_from_train_mask(train_graph)
            self.detect_info["hard_removed_train_nodes"] = int(self.hard_removed_train_nodes)
            if verbose:
                print(f"[HeteroGuard] hard removed train nodes: {self.hard_removed_train_nodes}")

        model = self._create_model(train_graph)
        trainer = DefenseTrainer(model=model, data=train_graph, target_type=self.target_type, device=self.device)

        if hard_remove_suspicious and not use_clean_graph and self.hard_removed_train_nodes > 0:
            current_train_count = int(train_graph[self.target_type].train_mask.sum().item())
            training_scores = torch.zeros(current_train_count, dtype=torch.float)
        else:
            training_scores = self.suspicious_scores

        if use_trigger_unlearning:
            if attacker is None:
                raise ValueError("use_trigger_unlearning=True requires attacker.")
            if target_class is None:
                target_class = getattr(attacker, "target_class", 0)
            info = trainer.fit_weighted_unlearn(
                suspicious_scores=training_scores,
                attacker=attacker,
                target_class=int(target_class),
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                min_weight=min_weight,
                max_downweight=max_downweight,
                unlearn_lambda=unlearn_lambda,
                unlearn_samples=unlearn_samples,
                target_suppression=target_suppression,
                unlearn_exclude_target=unlearn_exclude_target,
                verbose=verbose,
            )
        else:
            info = trainer.fit_weighted(
                suspicious_scores=training_scores,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                min_weight=min_weight,
                max_downweight=max_downweight,
                verbose=verbose,
            )
        info["hard_removed_train_nodes"] = int(self.hard_removed_train_nodes)
        info["use_trigger_unlearning"] = bool(use_trigger_unlearning)
        self.defense_model = trainer.model
        return info

    def get_model(self):
        if self.defense_model is None:
            raise RuntimeError("Defense model has not been trained. Call train_defense() first.")
        return self.defense_model

    @torch.no_grad()
    def predict(self, data: Optional[HeteroData] = None):
        model = self.get_model()
        model.eval()
        if data is None:
            data = self.purified_data if self.purified_data is not None else self.poison_data
        return model(data.to(self.device))

    @torch.no_grad()
    def clean_accuracy(self, data: Optional[HeteroData] = None):
        if data is None:
            data = self.clean_data
        model = self.get_model()
        model.eval()
        data = data.to(self.device)
        logits = model(data)[self.target_type]
        mask = data[self.target_type].test_mask
        y = data[self.target_type].y
        pred = logits[mask].argmax(dim=1)
        return float((pred == y[mask]).float().mean().item())
