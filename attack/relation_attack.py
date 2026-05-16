# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch_geometric.data import HeteroData

from .base import BaseHetAttack, EdgeType


class HetRelationBA(BaseHetAttack):
    """
    Relation-Hybrid attack.

    Main heterogeneous attack:
        auxiliary trigger nodes --existing hetero relation--> target nodes,
        optional auxiliary clique,
        optional target feature boost.

    Default is hybrid because pure relation-only triggers were empirically weak.
    """

    attack_name = "relation"

    def __init__(
        self,
        target_node_type: str,
        num_classes: int,
        target_class: int = 0,
        poison_rate: float = 0.1,
        trigger_size: int = 10,
        seed: int = 42,
        relation_edge_type: Optional[EdgeType] = None,
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
            flip_label=True,
            exclude_target_class=True,
            target_feature_boost=target_feature_boost,
            target_feature_strength=target_feature_strength,
            **kwargs,
        )

        self.relation_edge_type = relation_edge_type
        self.aux_node_type: Optional[str] = None
        self.use_aux_clique = bool(use_aux_clique)
        self.aux_feature_strength = float(aux_feature_strength)

        self.shared_aux_triggers = torch.empty(0, dtype=torch.long)
        self.aux_prototype_features: Optional[torch.Tensor] = None

    def _get_or_create_aux_features(self, data: HeteroData) -> torch.Tensor:
        if self.aux_node_type is None:
            raise ValueError("aux_node_type is not set.")

        if self.aux_prototype_features is not None:
            return self.aux_prototype_features.to(data[self.aux_node_type].x.device)

        proto = self._make_trigger_features(
            data=data,
            node_type=self.aux_node_type,
            num_triggers=self.trigger_size,
            target_like=False,
            high_variance_boost=True,
            strength=self.aux_feature_strength,
        ).detach()

        self.aux_prototype_features = proto.detach().cpu()
        return proto

    def _append_aux_trigger_group(self, data: HeteroData) -> torch.Tensor:
        if self.aux_node_type is None:
            raise ValueError("aux_node_type is not set.")
        proto = self._get_or_create_aux_features(data)
        return self._append_nodes(
            data=data,
            node_type=self.aux_node_type,
            new_x=proto.clone(),
            y_value=None,
        )

    def _connect_relation(self, data: HeteroData, aux_triggers: torch.Tensor, target_nodes: torch.Tensor):
        if self.relation_edge_type is None:
            raise ValueError("relation_edge_type is not set.")
        self._connect_aux_triggers_to_targets(data, self.relation_edge_type, aux_triggers, target_nodes)
        if self.use_aux_clique:
            self._add_trigger_clique(data, self.aux_node_type, aux_triggers, rel_name="trigger_link")

    def poison(self, clean_data: HeteroData) -> HeteroData:
        data = self._clone_data(clean_data)
        self._record_basic_metadata(data)

        relation = self._find_incoming_relation(data, self.relation_edge_type)
        self.relation_edge_type = relation
        self.aux_node_type = relation[0]

        poison_nodes = self._select_poison_nodes(data)
        self.poisoned_node_indices = poison_nodes.cpu()

        _ = self._get_or_create_aux_features(data)
        if self.target_feature_boost:
            _ = self._get_or_create_target_trigger_vector(data)

        aux_triggers = self._append_aux_trigger_group(data)
        self.shared_aux_triggers = aux_triggers.cpu()
        self.trigger_node_indices[self.aux_node_type] = aux_triggers.cpu()

        self._connect_relation(data, aux_triggers, poison_nodes)
        self._apply_target_feature_trigger(data, poison_nodes)
        self._flip_labels(data, poison_nodes)

        self.attack_metadata.update({
            "poisoned_nodes": int(poison_nodes.numel()),
            "trigger_nodes": int(aux_triggers.numel()),
            "trigger_type": "relation_hybrid" if self.target_feature_boost else "relation_pure",
            "relation_edge_type": str(relation),
            "aux_node_type": self.aux_node_type,
            "use_aux_clique": bool(self.use_aux_clique),
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

        if self.relation_edge_type is None:
            relation = self._find_incoming_relation(injected, None)
            self.relation_edge_type = relation
            self.aux_node_type = relation[0]
        else:
            relation = self.relation_edge_type
            self.aux_node_type = relation[0]

        n_target = injected[self.target_node_type].x.size(0)
        valid = inject_nodes[inject_nodes < n_target]
        if valid.numel() == 0:
            return injected

        if (
            self.shared_aux_triggers.numel() > 0
            and int(self.shared_aux_triggers.max().item()) < injected[self.aux_node_type].x.size(0)
        ):
            aux_triggers = self.shared_aux_triggers.clone()
        else:
            aux_triggers = self._append_aux_trigger_group(injected)
            self.shared_aux_triggers = aux_triggers.cpu()
            self.trigger_node_indices[self.aux_node_type] = aux_triggers.cpu()

        self._connect_relation(injected, aux_triggers, valid)
        self._apply_target_feature_trigger(injected, valid)
        return injected
