# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Union

import torch
from torch_geometric.data import HeteroData

from .base import BaseHetAttack


class HetCBA(BaseHetAttack):
    """
    CBA-Hybrid: community/boundary-oriented backdoor attack.

    Selects structurally boundary-like target nodes using relation-degree entropy,
    then applies a shared hybrid trigger.
    """

    attack_name = "cba"

    def __init__(
        self,
        target_node_type: str,
        num_classes: int,
        target_class: int = 0,
        poison_rate: float = 0.1,
        trigger_size: int = 10,
        seed: int = 42,
        use_clique: bool = True,
        target_feature_boost: bool = True,
        target_feature_strength: float = 4.0,
        trigger_feature_strength: float = 6.0,
        **kwargs,
    ):
        super().__init__(
            target_node_type=target_node_type,
            num_classes=num_classes,
            target_class=target_class,
            poison_rate=poison_rate,
            trigger_size=trigger_size,
            seed=seed,
            flip_label=True,
            exclude_target_class=True,
            target_feature_boost=target_feature_boost,
            target_feature_strength=target_feature_strength,
            **kwargs,
        )
        self.use_clique = bool(use_clique)
        self.trigger_feature_strength = float(trigger_feature_strength)
        self.shared_triggers = torch.empty(0, dtype=torch.long)
        self.prototype_features = None
        self.boundary_scores = None

    def _compute_boundary_scores(self, data: HeteroData) -> torch.Tensor:
        n = data[self.target_node_type].x.size(0)
        relation_degrees = []

        for et in data.edge_types:
            src_type, _, dst_type = et
            if self.target_node_type not in (src_type, dst_type):
                continue

            ei = data[et].edge_index.detach().cpu()
            deg = torch.zeros(n, dtype=torch.float)

            if src_type == self.target_node_type:
                idx = ei[0].long()
            else:
                idx = ei[1].long()

            valid = idx[idx < n]
            if valid.numel() > 0:
                deg.scatter_add_(0, valid, torch.ones_like(valid, dtype=torch.float))
            relation_degrees.append(deg)

        if not relation_degrees:
            return torch.zeros(n, dtype=torch.float)

        deg_mat = torch.stack(relation_degrees, dim=1)
        total = deg_mat.sum(dim=1)
        prob = deg_mat / total.clamp(min=1.0).unsqueeze(1)
        entropy = -(prob * (prob + 1e-12).log()).sum(dim=1)
        return torch.log1p(total) + entropy

    def _get_or_create_prototype_features(self, data: HeteroData) -> torch.Tensor:
        if self.prototype_features is not None:
            return self.prototype_features.to(data[self.target_node_type].x.device)

        proto = self._make_trigger_features(
            data=data,
            node_type=self.target_node_type,
            num_triggers=self.trigger_size,
            target_like=True,
            high_variance_boost=True,
            strength=self.trigger_feature_strength,
        ).detach()
        self.prototype_features = proto.detach().cpu()
        return proto

    def _append_shared_triggers(self, data: HeteroData) -> torch.Tensor:
        proto = self._get_or_create_prototype_features(data)
        return self._append_nodes(
            data=data,
            node_type=self.target_node_type,
            new_x=proto.clone(),
            y_value=self.target_class,
        )

    def _connect(self, data: HeteroData, triggers: torch.Tensor, target_nodes: torch.Tensor):
        self._connect_target_triggers_to_nodes(data, triggers, target_nodes)
        if self.use_clique:
            self._add_trigger_clique(data, self.target_node_type, triggers, rel_name="trigger_link")

    def poison(self, clean_data: HeteroData) -> HeteroData:
        data = self._clone_data(clean_data)
        self._record_basic_metadata(data)

        scores = self._compute_boundary_scores(data)
        self.boundary_scores = scores

        poison_nodes = self._select_poison_nodes(data, prefer_boundary_scores=scores)
        self.poisoned_node_indices = poison_nodes.cpu()

        _ = self._get_or_create_prototype_features(data)
        if self.target_feature_boost:
            _ = self._get_or_create_target_trigger_vector(data)

        triggers = self._append_shared_triggers(data)
        self.shared_triggers = triggers.cpu()
        self.trigger_node_indices[self.target_node_type] = triggers.cpu()

        self._connect(data, triggers, poison_nodes)
        self._apply_target_feature_trigger(data, poison_nodes)
        self._flip_labels(data, poison_nodes)

        self.attack_metadata.update({
            "poisoned_nodes": int(poison_nodes.numel()),
            "trigger_nodes": int(triggers.numel()),
            "trigger_type": "cba_hybrid",
            "selection": "relation_degree_entropy_topk",
            "use_clique": bool(self.use_clique),
            "target_feature_boost": bool(self.target_feature_boost),
        })
        return data

    def inject_trigger(
        self,
        data: HeteroData,
        inject_nodes: Union[torch.Tensor, Sequence[int]],
    ) -> HeteroData:
        injected = self._clone_data(data)
        inject_nodes = self._to_long_tensor(inject_nodes)

        if inject_nodes.numel() == 0:
            return injected

        n_target = injected[self.target_node_type].x.size(0)
        valid = inject_nodes[inject_nodes < n_target]
        if valid.numel() == 0:
            return injected

        if self.shared_triggers.numel() > 0 and int(self.shared_triggers.max().item()) < n_target:
            triggers = self.shared_triggers.clone()
        else:
            triggers = self._append_shared_triggers(injected)
            self.shared_triggers = triggers.cpu()
            self.trigger_node_indices[self.target_node_type] = triggers.cpu()

        self._connect(injected, triggers, valid)
        self._apply_target_feature_trigger(injected, valid)
        return injected
