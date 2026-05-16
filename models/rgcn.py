# -*- coding: utf-8 -*-
"""
RGCN backbone for heterogeneous node classification.

Unlike the old version, this class accepts a PyG HeteroData object directly.
It internally projects each node type to a shared hidden dimension, constructs
a temporary homogeneous graph, applies RGCNConv, and maps outputs back to
node-type dictionaries.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
    """
    Relational GCN over a homogeneous view of HeteroData.

    Best suited as a lightweight relation-aware baseline.
    """

    arch_version = 10

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        out_channels: int,
        hidden_dim: int = 128,
        num_bases: int = 8,
        dropout: float = 0.5,
        num_layers: int = 2,
        target_node_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels_dict = dict(in_channels_dict)
        self.out_channels = int(out_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_bases = int(num_bases)
        self.dropout = float(dropout)
        self.num_layers = max(1, int(num_layers))
        self.target_node_type = target_node_type

        self.proj = None
        self.convs = None
        self.classifiers = None
        self.edge_type_to_id = None
        self._node_types = None
        self._edge_types = None

    def _build_layers(self, metadata):
        node_types, edge_types = metadata
        self._node_types = list(node_types)
        self._edge_types = list(edge_types)
        self.edge_type_to_id = {
            edge_type: idx for idx, edge_type in enumerate(self._edge_types)
        }

        self.proj = nn.ModuleDict()
        for node_type in self._node_types:
            if node_type not in self.in_channels_dict:
                raise KeyError(
                    f"Missing input dimension for node type '{node_type}'."
                )
            self.proj[node_type] = nn.Linear(
                self.in_channels_dict[node_type], self.hidden_dim
            )

        num_relations = max(1, len(self._edge_types))
        num_bases = min(self.num_bases, num_relations)

        self.convs = nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(
                RGCNConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    num_relations=num_relations,
                    num_bases=num_bases,
                )
            )

        self.classifiers = nn.ModuleDict({
            node_type: nn.Linear(self.hidden_dim, self.out_channels)
            for node_type in self._node_types
        })

        self.add_module("proj", self.proj)
        self.add_module("convs", self.convs)
        self.add_module("classifiers", self.classifiers)

    def _check_metadata(self, data):
        metadata = (list(data.node_types), list(data.edge_types))
        if self.convs is None:
            self._build_layers(metadata)
            # Layers are lazily created after model.to(device) may have been called.
            # Move newly-created modules to the same device as input node features.
            first_x = next(iter(data.x_dict.values()))
            self.to(first_x.device)
        elif self._node_types != list(data.node_types) or self._edge_types != list(data.edge_types):
            raise ValueError(
                "RGCN was initialized with one graph metadata but received another. "
                "Create a new model after poisoning if new edge types were added."
            )

    def _to_homogeneous_view(self, data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Tuple[int, int]]]:
        """
        Constructs:
            x_homo: [sum_num_nodes, hidden_dim]
            edge_index_homo: [2, num_edges]
            edge_type: [num_edges]
            node_slices: node_type -> (start, end)
        """
        x_list = []
        node_slices = {}
        offset = 0

        for node_type in self._node_types:
            x = self.proj[node_type](data[node_type].x.float())
            n = x.size(0)
            node_slices[node_type] = (offset, offset + n)
            x_list.append(x)
            offset += n

        x_homo = torch.cat(x_list, dim=0) if x_list else torch.empty(0, self.hidden_dim)

        edge_indices = []
        edge_types = []

        for edge_type in self._edge_types:
            src_type, _, dst_type = edge_type
            ei = data[edge_type].edge_index
            if ei.numel() == 0:
                continue

            src_start = node_slices[src_type][0]
            dst_start = node_slices[dst_type][0]

            e = ei.clone()
            e[0] = e[0] + src_start
            e[1] = e[1] + dst_start

            edge_indices.append(e)

            rel_id = self.edge_type_to_id[edge_type]
            edge_types.append(
                torch.full(
                    (e.size(1),),
                    rel_id,
                    dtype=torch.long,
                    device=e.device,
                )
            )

        if edge_indices:
            edge_index_homo = torch.cat(edge_indices, dim=1).contiguous()
            edge_type_homo = torch.cat(edge_types, dim=0).contiguous()
        else:
            device = x_homo.device
            edge_index_homo = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type_homo = torch.empty((0,), dtype=torch.long, device=device)

        return x_homo, edge_index_homo, edge_type_homo, node_slices

    def encode(self, data) -> Dict[str, torch.Tensor]:
        self._check_metadata(data)

        x, edge_index, edge_type, node_slices = self._to_homogeneous_view(data)

        for layer_idx, conv in enumerate(self.convs):
            h = conv(x, edge_index, edge_type)
            x = x + F.dropout(h, p=self.dropout, training=self.training)
            if layer_idx < len(self.convs) - 1:
                x = F.relu(x)

        out_dict = {}
        for node_type, (start, end) in node_slices.items():
            out_dict[node_type] = x[start:end]

        return out_dict

    def forward(self, data, return_node_embeddings: bool = False):
        emb_dict = self.encode(data)
        if return_node_embeddings:
            return emb_dict

        return {
            node_type: self.classifiers[node_type](emb)
            for node_type, emb in emb_dict.items()
        }

    def get_target_embeddings(self, data, target_node_type: Optional[str] = None):
        target = target_node_type or self.target_node_type
        if target is None:
            raise ValueError("target_node_type must be provided.")
        return self.encode(data)[target]

    def reset_parameters(self):
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
