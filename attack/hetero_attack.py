# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Optional

from .feature_attack import HetFeatureAttack
from .sba_attack import HetSBA
from .uba_attack import HetUBA
from .relation_attack import HetRelationBA
from .clean_label_attack import HetCleanLabelBA
from .community_attack import HetCBA
from .grad_attack import HetGradBA


MAIN_ATTACKS = [
    "feature",
    "sba",
    "uba",
    "relation",
    "clean_label",
    "cba",
]

ATTACK_REGISTRY = {
    "feature": HetFeatureAttack,
    "feat": HetFeatureAttack,
    "feature_attack": HetFeatureAttack,
    "featureba": HetFeatureAttack,

    "sba": HetSBA,
    "sba_hybrid": HetSBA,
    "hetsba": HetSBA,

    "uba": HetUBA,
    "uba_hybrid": HetUBA,
    "hetuba": HetUBA,

    "relation": HetRelationBA,
    "relation_hybrid": HetRelationBA,
    "relation_pure": HetRelationBA,
    "rel": HetRelationBA,
    "metapath": HetRelationBA,
    "meta_path": HetRelationBA,
    "hetrelationba": HetRelationBA,

    "clean_label": HetCleanLabelBA,
    "clean": HetCleanLabelBA,
    "cl": HetCleanLabelBA,
    "cleanlabel": HetCleanLabelBA,
    "clean_label_hybrid": HetCleanLabelBA,

    "cba": HetCBA,
    "cba_hybrid": HetCBA,
    "community": HetCBA,
    "hetcba": HetCBA,

    "grad": HetGradBA,
    "gradient": HetGradBA,
    "gba": HetGradBA,
    "gcba": HetGradBA,
    "hetgradba": HetGradBA,
}


def normalize_attack_name(attack_name: str) -> str:
    key = str(attack_name).strip().lower().replace("-", "_")
    if key not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack_name={attack_name}. Supported: {get_attack_names()}")
    return key


def get_attack_names(main_only: bool = True):
    if main_only:
        return list(MAIN_ATTACKS)
    return sorted(ATTACK_REGISTRY.keys())


def create_attacker(
    attack_name: str,
    target_node_type: Optional[str] = None,
    num_classes: Optional[int] = None,
    attacker_kwargs: Optional[Dict] = None,
    **kwargs,
):
    """
    Flexible factory.

    Supports:
        create_attacker("uba", target_node_type="paper", num_classes=3, poison_rate=0.2)
        create_attacker("uba", "paper", 3, {"poison_rate": 0.2})
    """
    merged = {}
    if attacker_kwargs is not None:
        merged.update(attacker_kwargs)
    merged.update(kwargs)

    if target_node_type is None:
        target_node_type = merged.pop("target_node_type", None)
    if num_classes is None:
        num_classes = merged.pop("num_classes", None)

    if target_node_type is None:
        raise ValueError("target_node_type must be provided.")
    if num_classes is None:
        raise ValueError("num_classes must be provided.")

    # Convenience: relation_pure disables target feature boost.
    key = normalize_attack_name(attack_name)
    if key == "relation_pure":
        merged["target_feature_boost"] = False
        key = "relation"
    elif key == "relation_hybrid":
        merged["target_feature_boost"] = True
        key = "relation"

    cls = ATTACK_REGISTRY[key]
    return cls(
        target_node_type=target_node_type,
        num_classes=int(num_classes),
        **merged,
    )


__all__ = [
    "HetFeatureAttack",
    "HetSBA",
    "HetUBA",
    "HetRelationBA",
    "HetCleanLabelBA",
    "HetCBA",
    "HetGradBA",
    "create_attacker",
    "get_attack_names",
    "ATTACK_REGISTRY",
    "MAIN_ATTACKS",
]
