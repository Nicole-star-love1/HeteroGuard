# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch_geometric.data import HeteroData

from .base import BaseHetAttack


class HetFeatureAttack(BaseHetAttack):
    """
    Feature backdoor attack.

    Dirty-label feature trigger:
        - modifies target-node features;
        - flips poisoned labels to target class;
        - uses the same feature vector during ASR injection.
    """

    attack_name = "feature"

    def __init__(
        self,
        target_node_type: str,
        num_classes: int,
        target_class: int = 0,
        poison_rate: float = 0.1,
        trigger_size: int = 5,
        trigger_strength: float = 6.0,
        seed: int = 42,
        **kwargs,
    ):
        # run_integrated.py passes common hybrid kwargs to all attacks.
        # FeatureAttack is already a pure feature-trigger attack, so remove
        # structural/relation-hybrid options that would otherwise be forwarded
        # twice or become irrelevant.
        kwargs.pop("target_feature_boost", None)
        kwargs.pop("target_feature_strength", None)
        kwargs.pop("aux_feature_strength", None)
        kwargs.pop("use_aux_clique", None)

        super().__init__(
            target_node_type=target_node_type,
            num_classes=num_classes,
            target_class=target_class,
            poison_rate=poison_rate,
            trigger_size=max(1, trigger_size),
            seed=seed,
            flip_label=True,
            exclude_target_class=True,
            target_feature_boost=False,
            **kwargs,
        )
        self.trigger_strength = float(trigger_strength)
        self.trigger_vector: Optional[torch.Tensor] = None

    def _build_trigger_vector(self, data: HeteroData) -> torch.Tensor:
        x = data[self.target_node_type].x.float()
        feat_dim = x.size(1)

        k = min(max(1, self.trigger_size), feat_dim)
        std = x.std(dim=0).clamp(min=1e-6)
        dims = torch.argsort(std, descending=True)[:k]

        trigger = torch.zeros(feat_dim, dtype=x.dtype, device=x.device)
        trigger[dims] = self.trigger_strength * std[dims]

        self.attack_metadata["trigger_dims"] = dims.cpu().tolist()
        self.attack_metadata["trigger_strength"] = self.trigger_strength
        return trigger

    def poison(self, clean_data: HeteroData) -> HeteroData:
        data = self._clone_data(clean_data)
        self._record_basic_metadata(data)

        poison_nodes = self._select_poison_nodes(data)
        self.poisoned_node_indices = poison_nodes.cpu()

        if self.trigger_vector is None:
            self.trigger_vector = self._build_trigger_vector(data).detach().cpu()

        x = data[self.target_node_type].x
        trigger = self.trigger_vector.to(x.device).to(x.dtype)
        x[poison_nodes.to(x.device)] = x[poison_nodes.to(x.device)] + trigger

        self._flip_labels(data, poison_nodes)
        self.trigger_node_indices[self.target_node_type] = torch.empty(0, dtype=torch.long)

        self.attack_metadata.update({
            "poisoned_nodes": int(poison_nodes.numel()),
            "trigger_type": "feature_vector",
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

        if self.trigger_vector is None:
            self.trigger_vector = self._build_trigger_vector(injected).detach().cpu()

        x = injected[self.target_node_type].x
        valid = inject_nodes[inject_nodes < x.size(0)]
        if valid.numel() == 0:
            return injected

        trigger = self.trigger_vector.to(x.device).to(x.dtype)
        x[valid.to(x.device)] = x[valid.to(x.device)] + trigger
        return injected
