# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch
from torch_geometric.data import HeteroData

from .base import BaseHetAttack


class HetSBA(BaseHetAttack):
    """
    SBA-Hybrid: per-node subgraph backdoor attack.

    Main-attack version:
        - each poisoned node gets an independent trigger group;
        - all trigger groups share the same feature template;
        - each trigger group has an internal clique;
        - optional target feature boost strengthens trigger consistency.
    """

    attack_name = "sba"

    def __init__(
        self,
        target_node_type: str,
        num_classes: int,
        target_class: int = 0,
        poison_rate: float = 0.1,
        trigger_size: int = 10,
        seed: int = 42,
        max_poison_nodes: int = 5000,
        use_clique: bool = True,
        target_feature_boost: bool = True,
        target_feature_strength: float = 4.0,
        trigger_feature_strength: float = 6.0,
        reuse_existing_trigger_for_injection: bool = True,
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
        self.max_poison_nodes = int(max_poison_nodes)
        self.use_clique = bool(use_clique)
        self.trigger_feature_strength = float(trigger_feature_strength)
        self.reuse_existing_trigger_for_injection = bool(reuse_existing_trigger_for_injection)

        self.poison_to_triggers: Dict[int, List[int]] = {}
        self.prototype_triggers: torch.Tensor = torch.empty(0, dtype=torch.long)
        self.prototype_features: Optional[torch.Tensor] = None

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

    def _append_trigger_group(self, data: HeteroData) -> torch.Tensor:
        proto = self._get_or_create_prototype_features(data)
        return self._append_nodes(
            data=data,
            node_type=self.target_node_type,
            new_x=proto.clone(),
            y_value=self.target_class,
        )

    def _connect_group(self, data: HeteroData, triggers: torch.Tensor, target_nodes: torch.Tensor):
        self._connect_target_triggers_to_nodes(data, triggers, target_nodes)
        if self.use_clique:
            self._add_trigger_clique(data, self.target_node_type, triggers, rel_name="trigger_link")

    def poison(self, clean_data: HeteroData) -> HeteroData:
        data = self._clone_data(clean_data)
        self._record_basic_metadata(data)

        train_idx = self._get_train_indices(data)
        k = max(1, int(round(len(train_idx) * self.poison_rate)))
        k = min(k, self.max_poison_nodes)

        poison_nodes = self._select_poison_nodes(data, k=k)
        self.poisoned_node_indices = poison_nodes.cpu()

        _ = self._get_or_create_prototype_features(data)
        if self.target_feature_boost:
            _ = self._get_or_create_target_trigger_vector(data)

        all_trigger_nodes = []
        self.poison_to_triggers = {}

        for node in poison_nodes.tolist():
            triggers = self._append_trigger_group(data)
            all_trigger_nodes.append(triggers)
            self.poison_to_triggers[int(node)] = triggers.cpu().tolist()
            self._connect_group(data, triggers, torch.tensor([int(node)], dtype=torch.long))

        all_triggers = torch.cat(all_trigger_nodes, dim=0).long() if all_trigger_nodes else torch.empty(0, dtype=torch.long)
        self.trigger_node_indices[self.target_node_type] = all_triggers.cpu()
        self.prototype_triggers = all_triggers[: self.trigger_size].cpu()

        self._apply_target_feature_trigger(data, poison_nodes)
        self._flip_labels(data, poison_nodes)

        self.attack_metadata.update({
            "poisoned_nodes": int(poison_nodes.numel()),
            "trigger_nodes": int(all_triggers.numel()),
            "trigger_type": "sba_hybrid",
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

        triggers = None
        if (
            self.reuse_existing_trigger_for_injection
            and self.prototype_triggers.numel() > 0
            and int(self.prototype_triggers.max().item()) < n_target
        ):
            triggers = self.prototype_triggers.clone()

        if triggers is None:
            triggers = self._append_trigger_group(injected)
            self.prototype_triggers = triggers.cpu()
            self.trigger_node_indices[self.target_node_type] = triggers.cpu()

        self._connect_group(injected, triggers, valid)
        self._apply_target_feature_trigger(injected, valid)
        return injected
