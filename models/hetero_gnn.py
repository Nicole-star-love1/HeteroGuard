# -*- coding: utf-8 -*-
"""
Model entry file for heterogeneous GNN backbones.

Backbones:
    - HAN
    - HGT
    - RGCN
    - HeteroSAGE

All models follow the same interface:
    model(data) -> Dict[node_type, logits]

Recommended usage in experiments:
    from models.hetero_gnn import create_model

    model = create_model(
        model_name=args.model,
        data=data,
        num_classes=num_classes,
        target_node_type=target_type,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
"""

from typing import Dict, Optional

from .han import HAN
from .hgt import HGT
from .rgcn import RGCN
from .heterosage import HeteroSAGE


MODEL_REGISTRY = {
    "han": HAN,
    "hgt": HGT,
    "rgcn": RGCN,
    "heterosage": HeteroSAGE,
    "sage": HeteroSAGE,
    "hetero_sage": HeteroSAGE,
}


def get_model_names():
    return ["HAN", "HGT", "RGCN", "HeteroSAGE"]


def normalize_model_name(model_name: str) -> str:
    key = str(model_name).strip().lower().replace("-", "_")
    if key == "hetero_graphsage":
        key = "heterosage"
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name={model_name}. "
            f"Supported models: {get_model_names()}"
        )
    return key


def infer_in_channels_dict(data) -> Dict[str, int]:
    return {
        node_type: int(data[node_type].x.size(1))
        for node_type in data.node_types
    }


def create_model(
    model_name: str = "HAN",
    data=None,
    in_channels_dict: Optional[Dict[str, int]] = None,
    out_channels: Optional[int] = None,
    num_classes: Optional[int] = None,
    target_node_type: Optional[str] = None,
    hidden_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.5,
    num_bases: int = 8,
    **kwargs,
):
    """
    Factory function for all heterogeneous backbones.

    Args:
        model_name:
            HAN / HGT / RGCN / HeteroSAGE
        data:
            HeteroData. Used to infer input dimensions.
        in_channels_dict:
            Optional manual input dimensions.
        out_channels / num_classes:
            Number of output classes.
        target_node_type:
            Optional target node type. Models still return logits for all node types.
    """
    key = normalize_model_name(model_name)

    if in_channels_dict is None:
        if data is None:
            raise ValueError("Either data or in_channels_dict must be provided.")
        in_channels_dict = infer_in_channels_dict(data)

    if out_channels is None:
        if num_classes is None:
            raise ValueError("Either out_channels or num_classes must be provided.")
        out_channels = int(num_classes)

    cls = MODEL_REGISTRY[key]

    common_kwargs = dict(
        in_channels_dict=in_channels_dict,
        out_channels=out_channels,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
        target_node_type=target_node_type,
    )

    if key in {"han", "hgt"}:
        common_kwargs["num_heads"] = num_heads

    if key == "rgcn":
        common_kwargs["num_bases"] = num_bases

    common_kwargs.update(kwargs)
    return cls(**common_kwargs)


def build_model_from_args(args, data, num_classes: int, target_node_type: str):
    """
    Convenience wrapper for step1_train.py / step2_defense.py.
    """
    model_name = getattr(args, "model", "HAN")
    hidden_dim = getattr(args, "hidden_dim", 128)
    num_heads = getattr(args, "num_heads", 4)
    num_layers = getattr(args, "num_layers", 2)
    dropout = getattr(args, "dropout", 0.5)
    num_bases = getattr(args, "num_bases", 8)

    return create_model(
        model_name=model_name,
        data=data,
        num_classes=num_classes,
        target_node_type=target_node_type,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        num_bases=num_bases,
    )


# Backward-compatible aliases.
__all__ = [
    "HAN",
    "HGT",
    "RGCN",
    "HeteroSAGE",
    "create_model",
    "build_model_from_args",
    "get_model_names",
    "infer_in_channels_dict",
]
