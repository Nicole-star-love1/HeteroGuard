# -*- coding: utf-8 -*-
"""
HeteroSAGE backbone for heterogeneous node classification.

This is a simple non-transformer, non-HAN baseline based on HeteroConv +
SAGEConv for each edge type.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroSAGE(nn.Module):
    """
    Heterogeneous GraphSAGE using one SAGEConv per edge type.

    Good as a lightweight non-attention baseline.
    """

    arch_version = 10

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        out_channels: int,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        num_layers: int = 2,
        target_node_type: Optional[str] = None,
        aggr: str = "sum",
        **kwargs,
    ):
        super().__init__()

        self.in_channels_dict = dict(in_channels_dict)
        self.out_channels = int(out_channels)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.num_layers = max(1, int(num_layers))
        self.target_node_type = target_node_type
        self.aggr = aggr

        self.proj = None
        self.convs = None
        self.norms = None
        self.classifiers = None
        self._metadata = None

    def _build_layers(self, metadata):
        node_types, edge_types = metadata
        self._metadata = metadata

        self.proj = nn.ModuleDict()
        for node_type in node_types:
            if node_type not in self.in_channels_dict:
                raise KeyError(
                    f"Missing input dimension for node type '{node_type}'."
                )
            self.proj[node_type] = nn.Linear(
                self.in_channels_dict[node_type], self.hidden_dim
            )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv(
                    in_channels=(self.hidden_dim, self.hidden_dim),
                    out_channels=self.hidden_dim,
                )
            self.convs.append(HeteroConv(conv_dict, aggr=self.aggr))
            self.norms.append(nn.ModuleDict({
                node_type: nn.LayerNorm(self.hidden_dim)
                for node_type in node_types
            }))

        self.classifiers = nn.ModuleDict({
            node_type: nn.Linear(self.hidden_dim, self.out_channels)
            for node_type in node_types
        })

        self.add_module("proj", self.proj)
        self.add_module("convs", self.convs)
        self.add_module("norms", self.norms)
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
                "HeteroSAGE was initialized with one graph metadata but received another. "
                "Create a new model after poisoning if new edge types were added."
            )

    def encode(self, data) -> Dict[str, torch.Tensor]:
        self._check_metadata(data)

        x_dict = {
            node_type: self.proj[node_type](data[node_type].x.float())
            for node_type in data.node_types
        }
        edge_index_dict = data.edge_index_dict

        for conv, norm_dict in zip(self.convs, self.norms):
            out_dict = conv(x_dict, edge_index_dict)

            next_x_dict = {}
            for node_type in x_dict:
                msg = out_dict[node_type] if node_type in out_dict else torch.zeros_like(x_dict[node_type])
                h = x_dict[node_type] + F.dropout(msg, p=self.dropout, training=self.training)
                h = norm_dict[node_type](h)
                h = F.relu(h)
                next_x_dict[node_type] = h

            x_dict = next_x_dict

        return x_dict

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
