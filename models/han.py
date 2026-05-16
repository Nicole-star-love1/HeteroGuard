# -*- coding: utf-8 -*-
"""
HAN backbone for heterogeneous node classification.

Unified interface:
    model(data) -> Dict[node_type, logits]
    model(data, return_node_embeddings=True) -> Dict[node_type, embeddings]

This file is intended to replace the old monolithic hetero_gnn.py HAN class.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv


class HAN(nn.Module):
    """
    HAN-style heterogeneous attention network.

    Notes:
        - This implementation uses PyG HANConv over the edge types in HeteroData.
        - It does not require manually precomputed meta-path adjacency matrices.
        - All node types are projected to a shared hidden dimension.
        - The forward output is always a dict so it is compatible with the existing
          training/evaluation code:
              out_dict = model(data)
              logits = out_dict[target_node_type]
    """

    arch_version = 10

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        out_channels: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.5,
        num_layers: int = 2,
        target_node_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels_dict = dict(in_channels_dict)
        self.out_channels = int(out_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.num_layers = max(1, int(num_layers))
        self.target_node_type = target_node_type

        self.proj = None
        self.convs = None
        self.classifiers = None
        self._metadata = None

    def _build_layers(self, metadata):
        node_types, edge_types = metadata
        self._metadata = metadata

        self.proj = nn.ModuleDict()
        for node_type in node_types:
            if node_type not in self.in_channels_dict:
                raise KeyError(
                    f"Missing input dimension for node type '{node_type}'. "
                    f"Known types: {list(self.in_channels_dict.keys())}"
                )
            self.proj[node_type] = nn.Linear(
                self.in_channels_dict[node_type], self.hidden_dim
            )

        self.convs = nn.ModuleList()
        projected_channels = {node_type: self.hidden_dim for node_type in node_types}
        for _ in range(self.num_layers):
            self.convs.append(
                HANConv(
                    in_channels=projected_channels,
                    out_channels=self.hidden_dim,
                    metadata=metadata,
                    heads=self.num_heads,
                    dropout=self.dropout,
                )
            )

        self.classifiers = nn.ModuleDict({
            node_type: nn.Linear(self.hidden_dim, self.out_channels)
            for node_type in node_types
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
        elif self._metadata != metadata:
            raise ValueError(
                "HAN was initialized with one graph metadata but received another. "
                "Create a new model after poisoning if new edge types were added."
            )

    def encode(self, data) -> Dict[str, torch.Tensor]:
        self._check_metadata(data)

        x_dict = {}
        for node_type in data.node_types:
            x = data[node_type].x
            x_dict[node_type] = self.proj[node_type](x.float())

        edge_index_dict = data.edge_index_dict

        for layer_idx, conv in enumerate(self.convs):
            out_dict = conv(x_dict, edge_index_dict)

            # HANConv may omit isolated node types. Preserve their previous states.
            next_x_dict = {}
            for node_type in x_dict:
                h = out_dict[node_type] if node_type in out_dict else x_dict[node_type]
                if layer_idx < len(self.convs) - 1:
                    h = F.elu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                next_x_dict[node_type] = h
            x_dict = next_x_dict

        return x_dict

    def forward(self, data, return_node_embeddings: bool = False):
        emb_dict = self.encode(data)
        if return_node_embeddings:
            return emb_dict

        logits_dict = {
            node_type: self.classifiers[node_type](emb)
            for node_type, emb in emb_dict.items()
        }
        return logits_dict

    def get_target_embeddings(self, data, target_node_type: Optional[str] = None):
        target = target_node_type or self.target_node_type
        if target is None:
            raise ValueError("target_node_type must be provided.")
        return self.encode(data)[target]

    def reset_parameters(self):
        # Layers are lazy-built. If not built yet, there is nothing to reset.
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
