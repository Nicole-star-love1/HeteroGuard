# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch_geometric.data import HeteroData

from .base import BaseHetAttack


class HetUBA(BaseHetAttack):
    """
    UBA-Hybrid: universal structural trigger attack.

    Main-attack version:
        - one shared trigger group for all poisoned nodes;
        - no silent trigger_size cap at 5;
        - internal trigger clique;
        - optional target feature boost.
    """

    attack_name = "uba"

    def __init__(
        self,
        target_node_type: str,
        num_classes: int,
        target_class: int = 0,
        poison_rate: float = 0.1,
        trigger_size: int = 10,
        seed: int = 42,
        max_trigger_size: int = 50,
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
            trigger_size=min(max(1, int(trigger_size)), int(max_trigger_size)),
            seed=seed,
            flip_label=True,
            exclude_target_class=True,
            target_feature_boost=target_feature_boost,
            target_feature_strength=target_feature_strength,
            **kwargs,
        )
        self.max_trigger_size = int(max_trigger_size)
        self.use_clique = bool(use_clique)
        self.trigger_feature_strength = float(trigger_feature_strength)

        self.shared_triggers = torch.empty(0, dtype=torch.long)
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

    def _append_shared_triggers(self, data: HeteroData) -> torch.Tensor:
        proto = self._get_or_create_prototype_features(data)
        return self._append_nodes(
            data=data,
            node_type=self.target_node_type,
            new_x=proto.clone(),
            y_value=self.target_class,
        )

    def _connect_shared(self, data: HeteroData, triggers: torch.Tensor, target_nodes: torch.Tensor):
        self._connect_target_triggers_to_nodes(data, triggers, target_nodes)
        if self.use_clique:
            self._add_trigger_clique(data, self.target_node_type, triggers, rel_name="trigger_link")

    def poison(self, clean_data: HeteroData) -> HeteroData:
        data = self._clone_data(clean_data)
        self._record_basic_metadata(data)

        poison_nodes = self._select_poison_nodes(data)
        self.poisoned_node_indices = poison_nodes.cpu()

        _ = self._get_or_create_prototype_features(data)
        if self.target_feature_boost:
            _ = self._get_or_create_target_trigger_vector(data)

        triggers = self._append_shared_triggers(data)
        self.shared_triggers = triggers.cpu()
        self.trigger_node_indices[self.target_node_type] = triggers.cpu()

        self._connect_shared(data, triggers, poison_nodes)
        self._apply_target_feature_trigger(data, poison_nodes)
        self._flip_labels(data, poison_nodes)

        self.attack_metadata.update({
            "poisoned_nodes": int(poison_nodes.numel()),
            "trigger_nodes": int(triggers.numel()),
            "trigger_type": "uba_hybrid",
            "use_clique": bool(self.use_clique),
            "target_feature_boost": bool(self.target_feature_boost),
            "max_trigger_size": int(self.max_trigger_size),
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

        self._connect_shared(injected, triggers, valid)
        self._apply_target_feature_trigger(injected, valid)
        return injected
