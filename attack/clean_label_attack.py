# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch_geometric.data import HeteroData

from .base import BaseHetAttack, EdgeType


class HetCleanLabelBA(BaseHetAttack):
    """
    CleanLabel-Hybrid attack.

    Labels are not flipped.
    Poisoned training nodes are selected from target class and receive trigger.
    At inference time, trigger is attached to non-target test nodes.

    Default trigger is relation-aware when possible; falls back to target-type
    shared trigger if no incoming relation exists.
    """

    attack_name = "clean_label"

    def __init__(
        self,
        target_node_type: str,
        num_classes: int,
        target_class: int = 0,
        poison_rate: float = 0.1,
        trigger_size: int = 10,
        seed: int = 42,
        relation_edge_type: Optional[EdgeType] = None,
        fallback_to_target_trigger: bool = True,
        use_aux_clique: bool = True,
        target_feature_boost: bool = True,
        aux_feature_strength: float = 6.0,
        target_feature_strength: float = 4.0,
        **kwargs,
    ):
        super().__init__(
            target_node_type=target_node_type,
            num_classes=num_classes,
            target_class=target_class,
            poison_rate=poison_rate,
            trigger_size=trigger_size,
            seed=seed,
            flip_label=False,
            exclude_target_class=False,
            target_feature_boost=target_feature_boost,
            target_feature_strength=target_feature_strength,
            **kwargs,
        )
        self.relation_edge_type = relation_edge_type
        self.fallback_to_target_trigger = bool(fallback_to_target_trigger)
        self.use_aux_clique = bool(use_aux_clique)
        self.aux_feature_strength = float(aux_feature_strength)

        self.trigger_mode: Optional[str] = None
        self.aux_node_type: Optional[str] = None
        self.trigger_node_type: Optional[str] = None
        self.shared_triggers = torch.empty(0, dtype=torch.long)
        self.prototype_features: Optional[torch.Tensor] = None

    def _choose_mode(self, data: HeteroData):
        try:
            relation = self._find_incoming_relation(data, self.relation_edge_type)
            self.relation_edge_type = relation
            self.aux_node_type = relation[0]
            self.trigger_node_type = self.aux_node_type
            self.trigger_mode = "relation"
        except Exception:
            if not self.fallback_to_target_trigger:
                raise
            self.trigger_node_type = self.target_node_type
            self.trigger_mode = "target"

    def _get_or_create_features(self, data: HeteroData) -> torch.Tensor:
        if self.trigger_node_type is None:
            raise ValueError("trigger_node_type is not set.")

        if self.prototype_features is not None:
            return self.prototype_features.to(data[self.trigger_node_type].x.device)

        proto = self._make_trigger_features(
            data=data,
            node_type=self.trigger_node_type,
            num_triggers=self.trigger_size,
            target_like=(self.trigger_node_type == self.target_node_type),
            high_variance_boost=True,
            strength=self.aux_feature_strength,
        ).detach()
        self.prototype_features = proto.detach().cpu()
        return proto

    def _append_trigger_group(self, data: HeteroData) -> torch.Tensor:
        if self.trigger_node_type is None:
            raise ValueError("trigger_node_type is not set.")
        proto = self._get_or_create_features(data)
        return self._append_nodes(
            data=data,
            node_type=self.trigger_node_type,
            new_x=proto.clone(),
            y_value=self.target_class if self.trigger_node_type == self.target_node_type else None,
        )

    def _connect(self, data: HeteroData, triggers: torch.Tensor, target_nodes: torch.Tensor):
        if self.trigger_mode == "relation":
            self._connect_aux_triggers_to_targets(data, self.relation_edge_type, triggers, target_nodes)
            if self.use_aux_clique:
                self._add_trigger_clique(data, self.aux_node_type, triggers, rel_name="trigger_link")
        else:
            self._connect_target_triggers_to_nodes(data, triggers, target_nodes)
            if self.use_aux_clique:
                self._add_trigger_clique(data, self.target_node_type, triggers, rel_name="trigger_link")

    def poison(self, clean_data: HeteroData) -> HeteroData:
        data = self._clone_data(clean_data)
        self._record_basic_metadata(data)
        self._choose_mode(data)

        poison_nodes = self._select_poison_nodes(
            data,
            from_target_class=True,
            exclude_target_class=False,
        )
        self.poisoned_node_indices = poison_nodes.cpu()

        _ = self._get_or_create_features(data)
        if self.target_feature_boost:
            _ = self._get_or_create_target_trigger_vector(data)

        triggers = self._append_trigger_group(data)
        self.shared_triggers = triggers.cpu()
        self.trigger_node_indices[self.trigger_node_type] = triggers.cpu()

        self._connect(data, triggers, poison_nodes)
        self._apply_target_feature_trigger(data, poison_nodes)
        # No label flipping.

        self.attack_metadata.update({
            "poisoned_nodes": int(poison_nodes.numel()),
            "trigger_nodes": int(triggers.numel()),
            "trigger_type": "clean_label_hybrid",
            "trigger_mode": self.trigger_mode,
            "label_flipped": False,
            "target_feature_boost": bool(self.target_feature_boost),
            "relation_edge_type": str(self.relation_edge_type) if self.relation_edge_type is not None else None,
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

        if self.trigger_mode is None:
            self._choose_mode(injected)

        n_target = injected[self.target_node_type].x.size(0)
        valid = inject_nodes[inject_nodes < n_target]
        if valid.numel() == 0:
            return injected

        if (
            self.shared_triggers.numel() > 0
            and self.trigger_node_type is not None
            and int(self.shared_triggers.max().item()) < injected[self.trigger_node_type].x.size(0)
        ):
            triggers = self.shared_triggers.clone()
        else:
            triggers = self._append_trigger_group(injected)
            self.shared_triggers = triggers.cpu()
            self.trigger_node_indices[self.trigger_node_type] = triggers.cpu()

        self._connect(injected, triggers, valid)
        self._apply_target_feature_trigger(injected, valid)
        return injected
