# -*- coding: utf-8 -*-
"""
Run one attack setting and evaluate multiple defenses on the same poisoned graph.

Defenses:
    none        : evaluate poisoned model directly.
    prune       : detect trigger nodes/edges, prune trigger edges, retrain.
    isolate     : detect suspicious target nodes, isolate their incident edges, retrain.
    retraining  : retrain from scratch on poisoned graph without any purification.
    hr          : HeteroGuard-HR = prune + hard suspicious-node removal.
    unlearn     : HeteroGuard-Unlearn = HR + trigger unlearning.

Example:
    python -m experiments.run_defense_suite \
      --dataset ACM --model HAN --attack uba \
      --poison_rate 0.2 --trigger_size 10 --epochs 200 \
      --defenses none,prune,isolate,retraining,hr,unlearn \
      --seed 1 --save_results
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData

from data.hetero_dataset import get_dataset
from attack.hetero_attack import create_attacker
from defense.hetero_guard import HeteroGuard
from defense.utils import align_clean_to_poison_metadata
from experiments.run_integrated import (
    build_attacker_kwargs,
    build_model,
    configure_reproducibility,
    train_model,
)
from experiments.utils import (
    measure_asr,
    measure_clean_acc,
    validate_edge_index,
    _jsonable,
)


DEFENSES = ["none", "prune", "isolate", "retraining", "hr", "unlearn"]


def clone_hetero_data(data: HeteroData) -> HeteroData:
    out = HeteroData()
    for node_type in data.node_types:
        for key, value in data[node_type].items():
            out[node_type][key] = value.clone() if isinstance(value, torch.Tensor) else deepcopy(value)
        if "x" in out[node_type] and out[node_type].x is not None:
            out[node_type].num_nodes = int(out[node_type].x.size(0))
        elif getattr(data[node_type], "num_nodes", None) is not None:
            out[node_type].num_nodes = int(data[node_type].num_nodes)
    for edge_type in data.edge_types:
        for key, value in data[edge_type].items():
            out[edge_type][key] = value.clone() if isinstance(value, torch.Tensor) else deepcopy(value)
    return out


def isolate_suspicious_target_nodes(data: HeteroData, target_type: str, suspicious_nodes: torch.Tensor) -> HeteroData:
    """
    Remove all message-passing edges incident to suspicious target nodes.
    Labels and train_mask are kept unchanged. This is a simple Isolate baseline.
    """
    out = clone_hetero_data(data)
    if suspicious_nodes is None or suspicious_nodes.numel() == 0:
        return out

    suspicious = set(suspicious_nodes.detach().cpu().long().tolist())

    for edge_type in list(out.edge_types):
        src_type, _, dst_type = edge_type
        if "edge_index" not in out[edge_type]:
            continue

        ei = out[edge_type].edge_index
        if ei.numel() == 0:
            continue

        keep = torch.ones(ei.size(1), dtype=torch.bool)

        if src_type == target_type:
            src_cpu = ei[0].detach().cpu().long().tolist()
            src_bad = torch.tensor([int(v) in suspicious for v in src_cpu], dtype=torch.bool)
            keep &= ~src_bad

        if dst_type == target_type:
            dst_cpu = ei[1].detach().cpu().long().tolist()
            dst_bad = torch.tensor([int(v) in suspicious for v in dst_cpu], dtype=torch.bool)
            keep &= ~dst_bad

        out[edge_type].edge_index = ei[:, keep.to(ei.device)].contiguous()

    return out


def eval_model(
    model,
    clean_data,
    poison_data,
    eval_graph,
    target_type,
    target_class,
    attacker,
    num_inject,
    device,
    seed,
) -> Dict:
    clean_acc = float(measure_clean_acc(model, eval_graph, target_type, device=device))
    asr, asr_clean_acc, natural_ratio = measure_asr(
        model=model,
        clean_data=clean_data,
        poison_data=poison_data,
        target_type=target_type,
        target_class=target_class,
        num_inject=num_inject,
        device=device,
        attacker=attacker,
        seed=seed,
        exclude_target_class=True,
    )
    asr = 0.0 if asr is None else float(asr)
    asr_clean_acc = 0.0 if asr_clean_acc is None else float(asr_clean_acc)
    natural_ratio = 0.0 if natural_ratio is None else float(natural_ratio)
    return {
        "defense_clean_acc": clean_acc,
        "defense_asr": asr,
        "defense_asr_clean_acc": asr_clean_acc,
        "natural_target_ratio": natural_ratio,
    }


def train_fresh_defense_model(args, graph, num_classes, target_type, device, seed_offset: int, save_best: str = "val_acc"):
    configure_reproducibility(args.seed + seed_offset)
    model = build_model(args, graph, num_classes, target_type, device)
    info = train_model(
        model=model,
        data=graph,
        target_type=target_type,
        epochs=args.defense_epochs,
        lr=args.defense_lr,
        weight_decay=args.defense_weight_decay,
        patience=args.patience,
        save_best=save_best,
        verbose=args.verbose,
    )
    return model, info


def make_guard(args, clean_data, poison_data, target_type, num_classes, device, seed_offset: int, true_poison_indices=None):
    configure_reproducibility(args.seed + seed_offset)
    guard = HeteroGuard(
        clean_data=clean_data,
        poison_data=poison_data,
        target_node_type=target_type,
        num_classes=num_classes,
        model_name=args.model,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_bases=args.num_bases,
        device=device,
    )
    pre_info = guard.pretrain_reference(
        epochs=args.pretrain_epochs,
        lr=args.pretrain_lr,
        weight_decay=args.defense_weight_decay,
        verbose=args.verbose,
    )
    suspicious, scores, det_metrics = guard.detect(
            true_poison_indices=true_poison_indices,
        target_class=args.target_class,
        top_k_ratio=args.detection_ratio,
        return_metrics=True,
        use_label_signal=args.use_label_signal,
    )
    return guard, pre_info, suspicious, scores, det_metrics


def run_defense_suite(args):
    configure_reproducibility(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    t0 = time.time()

    data, num_classes, target_type = get_dataset(args.dataset, data_dir=args.data_dir, validate=True, verbose=False)
    data = data.to(device)
    validate_edge_index(data, raise_error=True)

    # 1. Clean model.
    configure_reproducibility(args.seed + 101)
    clean_model = build_model(args, data, num_classes, target_type, device)
    clean_info = train_model(
        model=clean_model,
        data=data,
        target_type=target_type,
        epochs=args.clean_epochs if args.clean_epochs is not None else args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_best="val_acc",
        verbose=args.verbose,
    )
    clean_acc = float(measure_clean_acc(clean_model, data, target_type, device=device))

    # 2. Attack generation.
    configure_reproducibility(args.seed + 202)
    attacker_kwargs = build_attacker_kwargs(args)
    attacker = create_attacker(
        args.attack,
        target_node_type=target_type,
        num_classes=num_classes,
        attacker_kwargs=attacker_kwargs,
    )
    poison_data = attacker.poison(data)
    poison_data = poison_data.to(device)
    validate_edge_index(poison_data, raise_error=True)

    true_poison = attacker.get_poisoned_nodes()
    trigger_nodes = attacker.get_trigger_nodes()
    if isinstance(trigger_nodes, dict):
        n_trigger = sum(int(v.numel()) for v in trigger_nodes.values())
    elif isinstance(trigger_nodes, torch.Tensor):
        n_trigger = int(trigger_nodes.numel())
    else:
        n_trigger = 0

    # 3. Poisoned model.
    configure_reproducibility(args.seed + 303)
    poisoned_model = build_model(args, poison_data, num_classes, target_type, device)
    poison_info = train_model(
        model=poisoned_model,
        data=poison_data,
        target_type=target_type,
        epochs=args.poison_epochs if args.poison_epochs is not None else args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_best=args.poison_save_best,
        verbose=args.verbose,
    )
    backdoored_clean_acc = float(measure_clean_acc(poisoned_model, poison_data, target_type, device=device))
    poison_asr, poison_asr_clean_acc, natural_ratio = measure_asr(
        model=poisoned_model,
        clean_data=data,
        poison_data=poison_data,
        target_type=target_type,
        target_class=attacker.target_class,
        num_inject=args.num_inject,
        device=device,
        attacker=attacker,
        seed=args.seed,
        exclude_target_class=True,
    )
    poison_asr = 0.0 if poison_asr is None else float(poison_asr)
    poison_asr_clean_acc = 0.0 if poison_asr_clean_acc is None else float(poison_asr_clean_acc)
    natural_ratio = 0.0 if natural_ratio is None else float(natural_ratio)

    # 4. Clean-trigger diagnostic.
    configure_reproducibility(args.seed + 404)

    # Clean-trigger diagnostic must use the same metadata as poison_data.
    # Otherwise HAN/HGT-style models cache clean-graph metadata first, then fail
    # when ASR evaluation injects trigger edges with additional edge types.
    clean_probe_train_data = align_clean_to_poison_metadata(data, poison_data).to(device)

    clean_probe_model = build_model(args, clean_probe_train_data, num_classes, target_type, device)
    clean_probe_info = train_model(
        model=clean_probe_model,
        data=clean_probe_train_data,
        target_type=target_type,
        epochs=args.clean_probe_epochs if args.clean_probe_epochs is not None else args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_best="val_acc",
        verbose=args.verbose,
    )
    clean_trigger_asr, clean_trigger_clean_acc, clean_trigger_natural = measure_asr(
        model=clean_probe_model,
        clean_data=data,
        poison_data=poison_data,
        target_type=target_type,
        target_class=attacker.target_class,
        num_inject=args.num_inject,
        device=device,
        attacker=attacker,
        seed=args.seed,
        exclude_target_class=True,
    )
    clean_trigger_asr = 0.0 if clean_trigger_asr is None else float(clean_trigger_asr)
    clean_trigger_clean_acc = 0.0 if clean_trigger_clean_acc is None else float(clean_trigger_clean_acc)

    attack_metrics = {
        "clean_acc": clean_acc,
        "backdoored_clean_acc": backdoored_clean_acc,
        "clean_acc_drop": clean_acc - backdoored_clean_acc,
        "clean_trigger_asr": clean_trigger_asr,
        "clean_trigger_clean_acc": clean_trigger_clean_acc,
        "poisoned_asr": poison_asr,
        "poisoned_asr_clean_acc": poison_asr_clean_acc,
        "natural_target_ratio": natural_ratio,
        "learned_backdoor_gap": poison_asr - clean_trigger_asr,
        "n_poison_nodes": int(true_poison.numel()),
        "n_trigger_nodes": int(n_trigger),
    }

    selected = [x.strip().lower() for x in args.defenses.split(",") if x.strip()]
    if "all" in selected:
        selected = DEFENSES

    defenses = {}

    # none
    if "none" in selected:
        metrics = eval_model(
            model=poisoned_model,
            clean_data=data,
            poison_data=poison_data,
            eval_graph=poison_data,
            target_type=target_type,
            target_class=attacker.target_class,
            attacker=attacker,
            num_inject=args.num_inject,
            device=device,
            seed=args.seed,
        )
        metrics.update({
            "defense_name": "NoDefense",
            "detection_precision": None,
            "detection_recall": None,
            "detection_f1": None,
            "asr_drop": poison_asr - metrics["defense_asr"],
            "utility_drop": clean_acc - metrics["defense_clean_acc"],
        })
        defenses["none"] = metrics

    # retraining
    if "retraining" in selected:
        retrain_model, retrain_info = train_fresh_defense_model(
            args, poison_data, num_classes, target_type, device, seed_offset=501, save_best="val_acc"
        )
        metrics = eval_model(
            retrain_model, data, poison_data, poison_data, target_type,
            attacker.target_class, attacker, args.num_inject, device, args.seed
        )
        metrics.update({
            "defense_name": "Retraining",
            "train_info": retrain_info,
            "detection_precision": None,
            "detection_recall": None,
            "detection_f1": None,
            "asr_drop": poison_asr - metrics["defense_asr"],
            "utility_drop": clean_acc - metrics["defense_clean_acc"],
        })
        defenses["retraining"] = metrics

    # prune
    if "prune" in selected:
        guard, pre_info, suspicious, scores, det_metrics = make_guard(
            args, data, poison_data, target_type, num_classes, device, seed_offset=601,
        true_poison_indices=true_poison
)
        pruned_graph = guard.purify().to(device)
        pruned_model, pruned_info = train_fresh_defense_model(
            args, pruned_graph, num_classes, target_type, device, seed_offset=602, save_best="val_acc"
        )
        metrics = eval_model(
            pruned_model, data, poison_data, pruned_graph, target_type,
            attacker.target_class, attacker, args.num_inject, device, args.seed
        )
        metrics.update({
            "defense_name": "Prune",
            "train_info": pruned_info,
            "detection_precision": float(det_metrics["precision"]),
            "detection_recall": float(det_metrics["recall"]),
            "detection_f1": float(det_metrics["f1"]),
            "suspicious_nodes": int(suspicious.numel()),
            "asr_drop": poison_asr - metrics["defense_asr"],
            "utility_drop": clean_acc - metrics["defense_clean_acc"],
        })
        defenses["prune"] = metrics

    # isolate
    if "isolate" in selected:
        guard, pre_info, suspicious, scores, det_metrics = make_guard(
            args, data, poison_data, target_type, num_classes, device, seed_offset=701,
        true_poison_indices=true_poison
)
        isolated_graph = isolate_suspicious_target_nodes(poison_data, target_type, suspicious).to(device)
        isolated_model, isolated_info = train_fresh_defense_model(
            args, isolated_graph, num_classes, target_type, device, seed_offset=702, save_best="val_acc"
        )
        metrics = eval_model(
            isolated_model, data, poison_data, isolated_graph, target_type,
            attacker.target_class, attacker, args.num_inject, device, args.seed
        )
        metrics.update({
            "defense_name": "Isolate",
            "train_info": isolated_info,
            "detection_precision": float(det_metrics["precision"]),
            "detection_recall": float(det_metrics["recall"]),
            "detection_f1": float(det_metrics["f1"]),
            "suspicious_nodes": int(suspicious.numel()),
            "asr_drop": poison_asr - metrics["defense_asr"],
            "utility_drop": clean_acc - metrics["defense_clean_acc"],
        })
        defenses["isolate"] = metrics

    # HeteroGuard-HR
    if "hr" in selected:
        guard, pre_info, suspicious, scores, det_metrics = make_guard(
            args, data, poison_data, target_type, num_classes, device, seed_offset=801,
        true_poison_indices=true_poison
)
        info = guard.train_defense(
            epochs=args.defense_epochs,
            lr=args.defense_lr,
            weight_decay=args.defense_weight_decay,
            verbose=args.verbose,
            use_clean_graph=False,
            use_prune=True,
            min_weight=args.min_weight,
            max_downweight=args.max_downweight,
            hard_remove_suspicious=True,
            use_trigger_unlearning=False,
        )
        model = guard.get_model()
        eval_graph = guard.purified_data if guard.purified_data is not None else poison_data
        metrics = eval_model(
            model, data, poison_data, eval_graph, target_type,
            attacker.target_class, attacker, args.num_inject, device, args.seed
        )
        metrics.update({
            "defense_name": "HeteroGuard-HR",
            "train_info": info,
            "detection_precision": float(det_metrics["precision"]),
            "detection_recall": float(det_metrics["recall"]),
            "detection_f1": float(det_metrics["f1"]),
            "suspicious_nodes": int(suspicious.numel()),
            "hard_removed_train_nodes": int(info.get("hard_removed_train_nodes", 0)),
            "asr_drop": poison_asr - metrics["defense_asr"],
            "utility_drop": clean_acc - metrics["defense_clean_acc"],
        })
        defenses["hr"] = metrics

    # HeteroGuard-Unlearn
    if "unlearn" in selected:
        guard, pre_info, suspicious, scores, det_metrics = make_guard(
            args, data, poison_data, target_type, num_classes, device, seed_offset=901,
        true_poison_indices=true_poison
)
        info = guard.train_defense(
            epochs=args.defense_epochs,
            lr=args.defense_lr,
            weight_decay=args.defense_weight_decay,
            verbose=args.verbose,
            use_clean_graph=False,
            use_prune=True,
            min_weight=args.min_weight,
            max_downweight=args.max_downweight,
            hard_remove_suspicious=True,
            use_trigger_unlearning=True,
            attacker=attacker,
            target_class=attacker.target_class,
            unlearn_lambda=args.unlearn_lambda,
            unlearn_samples=args.unlearn_samples,
            target_suppression=args.target_suppression,
            unlearn_exclude_target=not args.include_target_in_unlearning,
        )
        model = guard.get_model()
        eval_graph = guard.purified_data if guard.purified_data is not None else poison_data
        metrics = eval_model(
            model, data, poison_data, eval_graph, target_type,
            attacker.target_class, attacker, args.num_inject, device, args.seed
        )
        metrics.update({
            "defense_name": "HeteroGuard-Unlearn",
            "train_info": info,
            "detection_precision": float(det_metrics["precision"]),
            "detection_recall": float(det_metrics["recall"]),
            "detection_f1": float(det_metrics["f1"]),
            "suspicious_nodes": int(suspicious.numel()),
            "hard_removed_train_nodes": int(info.get("hard_removed_train_nodes", 0)),
            "use_trigger_unlearning": True,
            "unlearn_lambda": args.unlearn_lambda,
            "unlearn_samples": args.unlearn_samples,
            "target_suppression": args.target_suppression,
            "asr_drop": poison_asr - metrics["defense_asr"],
            "utility_drop": clean_acc - metrics["defense_clean_acc"],
        })
        defenses["unlearn"] = metrics

    result = {
        "dataset": args.dataset,
        "model": args.model,
        "attack": args.attack,
        "seed": args.seed,
        "target_type": target_type,
        "num_classes": int(num_classes),
        "target_class": int(attacker.target_class),
        "poison_rate": float(args.poison_rate),
        "trigger_size": int(args.trigger_size),
        "attack_metrics": attack_metrics,
        "defenses": defenses,
        "attacker_info": attacker.get_attack_info() if hasattr(attacker, "get_attack_info") else {},
        "config": vars(args),
        "runtime_sec": time.time() - t0,
    }

    if args.save_results:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"defense_suite_{args.dataset}_{args.model}_{args.attack}_r{args.poison_rate:.3f}_seed{args.seed}_{ts}.json"
        path.write_text(json.dumps(_jsonable(result), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SAVED] {path}")

    print("=" * 80)
    print(f"Defense suite summary | {args.dataset} {args.model} {args.attack} seed={args.seed}")
    print("=" * 80)
    print(f"Clean={clean_acc:.4f}, BA={backdoored_clean_acc:.4f}, PoisonASR={poison_asr:.4f}, CleanTrigASR={clean_trigger_asr:.4f}, Gap={poison_asr-clean_trigger_asr:+.4f}")
    for k, v in defenses.items():
        print(f"{v['defense_name']:20s} | Clean={v['defense_clean_acc']:.4f}, ASR={v['defense_asr']:.4f}, Drop={v['asr_drop']:+.4f}")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Run six-defense suite on one heterogeneous backdoor setting.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results_defense_suite")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--dataset", type=str, default="ACM")
    parser.add_argument("--model", type=str, default="HAN")
    parser.add_argument("--attack", type=str, default="uba")

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_bases", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--clean_epochs", type=int, default=None)
    parser.add_argument("--poison_epochs", type=int, default=None)
    parser.add_argument("--clean_probe_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--poison_save_best", type=str, default="last", choices=["last", "train_loss", "val_acc"])

    parser.add_argument("--poison_rate", type=float, default=0.2)
    parser.add_argument("--trigger_size", type=int, default=10)
    parser.add_argument("--target_class", type=int, default=0)
    parser.add_argument("--trigger_strength", type=float, default=3.0)
    parser.add_argument("--surrogate_epochs", type=int, default=30)
    parser.add_argument("--num_inject", type=int, default=200)

    parser.add_argument("--relation_mode", type=str, default="hybrid", choices=["pure", "hybrid"])
    parser.add_argument("--target_feature_strength", type=float, default=4.0)
    parser.add_argument("--aux_feature_strength", type=float, default=6.0)
    parser.add_argument("--no_aux_clique", action="store_true")

    parser.add_argument("--defenses", type=str, default="all", help="Comma list: none,prune,isolate,retraining,hr,unlearn or all")

    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--pretrain_lr", type=float, default=0.005)
    parser.add_argument("--defense_epochs", type=int, default=100)
    parser.add_argument("--defense_lr", type=float, default=0.005)
    parser.add_argument("--defense_weight_decay", type=float, default=1e-4)
    parser.add_argument("--detection_ratio", type=float, default=0.2)
    parser.add_argument("--use_label_signal", action="store_true")
    parser.add_argument("--min_weight", type=float, default=0.0)
    parser.add_argument("--max_downweight", type=float, default=1.0)

    parser.add_argument("--unlearn_lambda", type=float, default=1.0)
    parser.add_argument("--unlearn_samples", type=int, default=256)
    parser.add_argument("--target_suppression", type=float, default=0.2)
    parser.add_argument("--include_target_in_unlearning", action="store_true")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_defense_suite(args)
