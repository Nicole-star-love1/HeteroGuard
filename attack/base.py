# -*- coding: utf-8 -*-
"""
Enhanced base utilities for heterogeneous graph backdoor attacks.

Design contract:
    poison(clean_data) -> poisoned training graph
    inject_trigger(data, inject_nodes) -> ASR evaluation graph

All attacks are designed for PyG HeteroData.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import HeteroData


EdgeType = Tuple[str, str, str]


class BaseHetAttack:
    attack_name = "base"

    def __init__(
        self,
        target_node_type: str,
        num_classes: int,
        target_class: int = 0,
        poison_rate: float = 0.1,
        trigger_size: int = 5,
        seed: int = 42,
        flip_label: bool = True,
        exclude_target_class: bool = True,
        feature_noise_std: float = 0.05,
        target_feature_boost: bool = False,
        target_feature_strength: float = 4.0,
        **kwargs,
    ):
        self.target_node_type = target_node_type
        self.num_classes = int(num_classes)
        self.target_class = int(target_class)
        self.poison_rate = float(poison_rate)
        self.trigger_size = max(1, int(trigger_size))
        self.seed = int(seed)
        self.flip_label = bool(flip_label)
        self.exclude_target_class = bool(exclude_target_class)
        self.feature_noise_std = float(feature_noise_std)

        self.target_feature_boost = bool(target_feature_boost)
        self.target_feature_strength = float(target_feature_strength)
        self.target_trigger_vector: Optional[torch.Tensor] = None

        self.rng = np.random.RandomState(self.seed)
        self.torch_gen = torch.Generator()
        self.torch_gen.manual_seed(self.seed)

        self.poisoned_node_indices: Optional[torch.Tensor] = None
        self.trigger_node_indices: Dict[str, torch.Tensor] = {}
        self.trigger_edge_types: List[EdgeType] = []
        self.attack_metadata: Dict = {}

    # ------------------------------------------------------------------
    # Required public interface
    # ------------------------------------------------------------------

    def poison(self, clean_data: HeteroData) -> HeteroData:
        raise NotImplementedError

    def inject_trigger(
        self,
        data: HeteroData,
        inject_nodes: Union[torch.Tensor, Sequence[int]],
    ) -> HeteroData:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_poisoned_nodes(self) -> torch.Tensor:
        if self.poisoned_node_indices is None:
            return torch.empty(0, dtype=torch.long)
        return self.poisoned_node_indices.detach().cpu().clone()

    def get_trigger_nodes(self, node_type: Optional[str] = None):
        if node_type is not None:
            return self.trigger_node_indices.get(
                node_type, torch.empty(0, dtype=torch.long)
            ).detach().cpu().clone()
        return {
            nt: idx.detach().cpu().clone()
            for nt, idx in self.trigger_node_indices.items()
        }

    def get_attack_info(self) -> Dict:
        return {
            "attack_name": self.attack_name,
            "target_node_type": self.target_node_type,
            "target_class": self.target_class,
            "poison_rate": self.poison_rate,
            "trigger_size": self.trigger_size,
            "poisoned_node_indices": self.get_poisoned_nodes().tolist(),
            "trigger_node_indices": {
                nt: idx.tolist() for nt, idx in self.get_trigger_nodes().items()
            },
            "trigger_edge_types": [str(et) for et in self.trigger_edge_types],
            "target_feature_boost": bool(self.target_feature_boost),
            "target_feature_strength": float(self.target_feature_strength),
            "metadata": deepcopy(self.attack_metadata),
        }

    # ------------------------------------------------------------------
    # Clone / validation
    # ------------------------------------------------------------------

    def _clone_data(self, data: HeteroData) -> HeteroData:
        out = HeteroData()

        for node_type in data.node_types:
            for key, value in data[node_type].items():
                out[node_type][key] = self._clone_value(value)
            if "x" in out[node_type] and out[node_type].x is not None:
                out[node_type].num_nodes = int(out[node_type].x.size(0))
            elif getattr(data[node_type], "num_nodes", None) is not None:
                out[node_type].num_nodes = int(data[node_type].num_nodes)

        for edge_type in data.edge_types:
            for key, value in data[edge_type].items():
                out[edge_type][key] = self._clone_value(value)

        return out

    @staticmethod
    def _clone_value(value):
        if isinstance(value, torch.Tensor):
            return value.clone().contiguous()
        return deepcopy(value)

    def _check_target(self, data: HeteroData):
        if self.target_node_type not in data.node_types:
            raise ValueError(f"Unknown target node type: {self.target_node_type}")
        for key in ["x", "y", "train_mask"]:
            if key not in data[self.target_node_type]:
                raise ValueError(f"{self.target_node_type} lacks required field: {key}")

    @staticmethod
    def _to_long_tensor(nodes: Union[torch.Tensor, Sequence[int]]) -> torch.Tensor:
        if isinstance(nodes, torch.Tensor):
            return nodes.detach().cpu().long().view(-1)
        return torch.tensor(list(nodes), dtype=torch.long)

    # ------------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------------

    def _get_train_indices(self, data: HeteroData) -> torch.Tensor:
        self._check_target(data)
        return torch.where(data[self.target_node_type].train_mask.detach().cpu())[0].long()

    def _select_poison_nodes(
        self,
        data: HeteroData,
        candidate_indices: Optional[torch.Tensor] = None,
        k: Optional[int] = None,
        exclude_target_class: Optional[bool] = None,
        from_target_class: bool = False,
        prefer_boundary_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._check_target(data)

        y = data[self.target_node_type].y.detach().cpu()
        train_idx = self._get_train_indices(data)

        if candidate_indices is None:
            candidates = train_idx
        else:
            candidates = self._to_long_tensor(candidate_indices)
            train_set = set(train_idx.tolist())
            candidates = torch.tensor(
                [int(i) for i in candidates.tolist() if int(i) in train_set],
                dtype=torch.long,
            )

        if candidates.numel() == 0:
            raise ValueError("No candidate nodes available for poisoning.")

        if from_target_class:
            filtered = candidates[y[candidates] == self.target_class]
        else:
            use_exclude = self.exclude_target_class if exclude_target_class is None else exclude_target_class
            if use_exclude:
                filtered = candidates[y[candidates] != self.target_class]
            else:
                filtered = candidates

        if filtered.numel() == 0:
            filtered = candidates

        if k is None:
            k = max(1, int(round(len(train_idx) * self.poison_rate)))
        k = min(int(k), int(filtered.numel()))

        if prefer_boundary_scores is not None:
            scores = prefer_boundary_scores.detach().cpu()
            top_pos = torch.argsort(scores[filtered], descending=True)[:k]
            return filtered[top_pos].long()

        perm = torch.randperm(filtered.numel(), generator=self.torch_gen)
        return filtered[perm[:k]].long()

    def _flip_labels(self, data: HeteroData, nodes: torch.Tensor):
        if not self.flip_label:
            return
        y = data[self.target_node_type].y
        nodes = nodes.to(y.device)
        y[nodes] = self.target_class

    # ------------------------------------------------------------------
    # Feature templates
    # ------------------------------------------------------------------

    def _class_feature_mean(
        self,
        data: HeteroData,
        node_type: str,
        class_id: Optional[int] = None,
    ) -> torch.Tensor:
        x = data[node_type].x.float()
        if node_type == self.target_node_type and "y" in data[node_type]:
            y = data[node_type].y.detach().cpu()
            cls = self.target_class if class_id is None else int(class_id)
            mask = y == cls
            if mask.sum().item() > 0:
                return x[mask.to(x.device)].mean(dim=0)
        return x.mean(dim=0)

    def _make_trigger_features(
        self,
        data: HeteroData,
        node_type: str,
        num_triggers: int,
        target_like: bool = True,
        high_variance_boost: bool = True,
        strength: float = 4.0,
    ) -> torch.Tensor:
        x = data[node_type].x.float()
        device = x.device
        dtype = x.dtype
        feat_dim = x.size(1)

        if target_like:
            center = self._class_feature_mean(data, node_type)
        else:
            center = x.mean(dim=0)

        proto = center.view(1, -1).repeat(num_triggers, 1)

        if high_variance_boost:
            std = x.std(dim=0).clamp(min=1e-6)
            k = min(max(1, num_triggers), feat_dim)
            dims = torch.argsort(std, descending=True)[:k]
            for i in range(num_triggers):
                dim = dims[i % k]
                proto[i, dim] = proto[i, dim] + float(strength) * std[dim]
            self.attack_metadata.setdefault("trigger_feature_dims", dims.detach().cpu().tolist())

        # CPU generator cannot create CUDA tensors directly.
        noise = torch.randn(
            num_triggers,
            feat_dim,
            generator=self.torch_gen,
            dtype=torch.float32,
        ).to(device=device, dtype=dtype) * self.feature_noise_std

        return proto + noise

    def _get_or_create_target_trigger_vector(self, data: HeteroData) -> torch.Tensor:
        if self.target_trigger_vector is not None:
            return self.target_trigger_vector.to(data[self.target_node_type].x.device)

        x = data[self.target_node_type].x.float()
        feat_dim = x.size(1)
        std = x.std(dim=0).clamp(min=1e-6)
        k = min(max(1, self.trigger_size), feat_dim)
        dims = torch.argsort(std, descending=True)[:k]

        trigger = torch.zeros(feat_dim, dtype=x.dtype, device=x.device)
        trigger[dims] = self.target_feature_strength * std[dims]

        self.target_trigger_vector = trigger.detach().cpu()
        self.attack_metadata["target_feature_boost_dims"] = dims.detach().cpu().tolist()
        return trigger

    def _apply_target_feature_trigger(self, data: HeteroData, target_nodes: torch.Tensor):
        if not self.target_feature_boost:
            return

        target_nodes = self._to_long_tensor(target_nodes)
        if target_nodes.numel() == 0:
            return

        x = data[self.target_node_type].x
        valid = target_nodes[target_nodes < x.size(0)]
        if valid.numel() == 0:
            return

        trigger = self._get_or_create_target_trigger_vector(data).to(device=x.device, dtype=x.dtype)
        x[valid.to(x.device)] = x[valid.to(x.device)] + trigger

    # ------------------------------------------------------------------
    # Node / edge editing
    # ------------------------------------------------------------------

    def _append_nodes(
        self,
        data: HeteroData,
        node_type: str,
        new_x: torch.Tensor,
        y_value: Optional[int] = None,
    ) -> torch.Tensor:
        old_n = int(data[node_type].x.size(0))
        new_x = new_x.to(data[node_type].x.device).to(data[node_type].x.dtype)
        data[node_type].x = torch.cat([data[node_type].x, new_x], dim=0).contiguous()
        data[node_type].num_nodes = int(data[node_type].x.size(0))

        new_n = int(new_x.size(0))
        new_idx = torch.arange(old_n, old_n + new_n, dtype=torch.long)

        if "y" in data[node_type] and data[node_type].y is not None:
            y = data[node_type].y
            fill = 0 if y_value is None else int(y_value)
            extra_y = torch.full((new_n,), fill, dtype=y.dtype, device=y.device)
            data[node_type].y = torch.cat([y, extra_y], dim=0).contiguous()

        for mask_name in ["train_mask", "val_mask", "test_mask", "hgb_train_mask", "hgb_test_mask"]:
            if mask_name in data[node_type] and data[node_type][mask_name] is not None:
                mask = data[node_type][mask_name]
                extra = torch.zeros(new_n, dtype=torch.bool, device=mask.device)
                data[node_type][mask_name] = torch.cat([mask, extra], dim=0).contiguous()

        return new_idx

    def _ensure_edge_type(self, data: HeteroData, edge_type: EdgeType):
        if edge_type not in data.edge_types or "edge_index" not in data[edge_type]:
            device = data[edge_type[0]].x.device
            data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    def _append_edges(self, data: HeteroData, edge_type: EdgeType, edges: torch.Tensor):
        if edges is None or edges.numel() == 0:
            self._ensure_edge_type(data, edge_type)
            return

        self._ensure_edge_type(data, edge_type)
        edges = edges.long().to(data[edge_type].edge_index.device)

        if edges.dim() != 2 or edges.size(0) != 2:
            raise ValueError(f"edges must have shape [2, E], got {tuple(edges.shape)}")

        src_type, _, dst_type = edge_type
        n_src = int(data[src_type].x.size(0))
        n_dst = int(data[dst_type].x.size(0))

        if edges.numel() > 0:
            if int(edges[0].min().item()) < 0 or int(edges[1].min().item()) < 0:
                raise ValueError(f"Negative edge index in {edge_type}")
            if int(edges[0].max().item()) >= n_src or int(edges[1].max().item()) >= n_dst:
                raise ValueError(
                    f"Edge index out of bounds for {edge_type}: "
                    f"src_max={int(edges[0].max().item())}/{n_src}, "
                    f"dst_max={int(edges[1].max().item())}/{n_dst}"
                )

        data[edge_type].edge_index = torch.cat([data[edge_type].edge_index, edges], dim=1).contiguous()

    @staticmethod
    def _cartesian_edges(src_nodes: torch.Tensor, dst_nodes: torch.Tensor) -> torch.Tensor:
        src_nodes = src_nodes.long().view(-1)
        dst_nodes = dst_nodes.long().view(-1)
        if src_nodes.numel() == 0 or dst_nodes.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long)
        src = src_nodes.repeat_interleave(dst_nodes.numel())
        dst = dst_nodes.repeat(src_nodes.numel())
        return torch.stack([src, dst], dim=0)

    def _target_backdoor_edge_types(self) -> Tuple[EdgeType, EdgeType]:
        t = self.target_node_type
        return (t, "backdoor", t), (t, "rev_backdoor", t)

    def _connect_target_triggers_to_nodes(
        self,
        data: HeteroData,
        trigger_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
    ):
        trigger_nodes = self._to_long_tensor(trigger_nodes)
        target_nodes = self._to_long_tensor(target_nodes)
        backdoor_et, rev_et = self._target_backdoor_edge_types()

        # trigger -> target and target -> trigger
        e1 = self._cartesian_edges(trigger_nodes, target_nodes)
        e2 = self._cartesian_edges(target_nodes, trigger_nodes)

        self._append_edges(data, backdoor_et, e1)
        self._append_edges(data, rev_et, e2)

        for et in [backdoor_et, rev_et]:
            if et not in self.trigger_edge_types:
                self.trigger_edge_types.append(et)

    def _add_trigger_clique(
        self,
        data: HeteroData,
        node_type: str,
        triggers: torch.Tensor,
        rel_name: str = "trigger_link",
    ):
        triggers = self._to_long_tensor(triggers)
        if triggers.numel() <= 1:
            return

        et = (node_type, rel_name, node_type)
        rev_et = (node_type, "rev_" + rel_name, node_type)

        clique = self._cartesian_edges(triggers, triggers)
        mask = clique[0] != clique[1]
        clique = clique[:, mask].contiguous()
        if clique.numel() == 0:
            return

        self._append_edges(data, et, clique)
        self._append_edges(data, rev_et, clique.flip(0).contiguous())

        for edge_type in [et, rev_et]:
            if edge_type not in self.trigger_edge_types:
                self.trigger_edge_types.append(edge_type)

    def _add_target_triggers(
        self,
        data: HeteroData,
        num_triggers: int,
        strength: float = 4.0,
    ) -> torch.Tensor:
        new_x = self._make_trigger_features(
            data=data,
            node_type=self.target_node_type,
            num_triggers=num_triggers,
            target_like=True,
            high_variance_boost=True,
            strength=strength,
        )
        return self._append_nodes(
            data=data,
            node_type=self.target_node_type,
            new_x=new_x,
            y_value=self.target_class,
        )

    # ------------------------------------------------------------------
    # Relation helpers
    # ------------------------------------------------------------------

    def _find_incoming_relation(
        self,
        data: HeteroData,
        preferred_edge_type: Optional[EdgeType] = None,
    ) -> EdgeType:
        if preferred_edge_type is not None:
            if preferred_edge_type not in data.edge_types:
                raise ValueError(f"preferred_edge_type={preferred_edge_type} not found.")
            if preferred_edge_type[2] != self.target_node_type:
                raise ValueError(
                    f"preferred_edge_type must point to {self.target_node_type}, got {preferred_edge_type}"
                )
            return preferred_edge_type

        candidates: List[EdgeType] = []
        for et in data.edge_types:
            src, _, dst = et
            if dst == self.target_node_type and src != self.target_node_type:
                candidates.append(et)

        if not candidates:
            raise ValueError(
                f"No incoming heterogeneous relation found for target node type {self.target_node_type}. "
                f"Make sure reverse edges are present."
            )

        candidates.sort(key=lambda et: data[et].edge_index.size(1), reverse=True)
        return candidates[0]

    def _add_aux_triggers(
        self,
        data: HeteroData,
        aux_node_type: str,
        num_triggers: int,
        strength: float = 6.0,
    ) -> torch.Tensor:
        new_x = self._make_trigger_features(
            data=data,
            node_type=aux_node_type,
            num_triggers=num_triggers,
            target_like=False,
            high_variance_boost=True,
            strength=strength,
        )
        return self._append_nodes(
            data=data,
            node_type=aux_node_type,
            new_x=new_x,
            y_value=None,
        )

    def _connect_aux_triggers_to_targets(
        self,
        data: HeteroData,
        relation_edge_type: EdgeType,
        aux_trigger_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
    ):
        src_type, _, dst_type = relation_edge_type
        if dst_type != self.target_node_type:
            raise ValueError(f"Relation edge must point to target type: {relation_edge_type}")

        aux_trigger_nodes = self._to_long_tensor(aux_trigger_nodes)
        target_nodes = self._to_long_tensor(target_nodes)

        edges = self._cartesian_edges(aux_trigger_nodes, target_nodes)
        self._append_edges(data, relation_edge_type, edges)

        if relation_edge_type not in self.trigger_edge_types:
            self.trigger_edge_types.append(relation_edge_type)

        reverse_candidates = [
            et for et in data.edge_types
            if et[0] == dst_type and et[2] == src_type
        ]
        if reverse_candidates:
            reverse_candidates.sort(
                key=lambda et: 0 if ("rev" in et[1].lower() or "reverse" in et[1].lower()) else 1
            )
            rev_et = reverse_candidates[0]
            self._append_edges(data, rev_et, edges.flip(0).contiguous())
            if rev_et not in self.trigger_edge_types:
                self.trigger_edge_types.append(rev_et)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _record_basic_metadata(self, data: HeteroData):
        self.attack_metadata.update({
            "node_types": list(data.node_types),
            "edge_types": [str(et) for et in data.edge_types],
            "target_node_type": self.target_node_type,
            "num_classes": self.num_classes,
        })
