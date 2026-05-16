# -*- coding: utf-8 -*-
"""
Experiment 7: Scalability study on OGBN-MAG / large heterogeneous graph.

Purpose:
    This runner is designed to evaluate whether HeteroGuard-HR and
    HeteroGuard-Unlearn can be executed on a large heterogeneous graph setting,
    while recording:
        - Defense ASR
        - Clean Acc
        - Defense time
        - Peak GPU memory
        - Number of nodes / edges processed

Recommended setting:
    Dataset: OGBN-MAG
    Model: HeteroSAGE or HGT
    Attacks: relation, uba
    Defenses: hr, unlearn
    Seeds: 1, 2, 3

Notes:
    1. Full OGBN-MAG can be heavy for full-batch HGT/HAN-style code.
       This runner supports --node_budget_per_type. If set to 0, it uses
       the full graph. If >0, it samples a reproducible induced heterogeneous
       subgraph from OGBN-MAG and reports the processed node/edge counts.
    2. If your project's data.hetero_dataset.get_dataset("OGBN-MAG") supports
       OGBN-MAG, this runner uses it. Otherwise it falls back to PyG's OGB_MAG.
    3. The code intentionally runs only two defenses: HeteroGuard-HR and
       HeteroGuard-Unlearn. This is a scalability study, not another baseline
       comparison table.

Example smoke:
    python -m experiments.run_scalability_suite \
      --dataset OGBN-MAG --model HeteroSAGE --attack relation \
      --poison_rate 0.05 --trigger_size 5 \
      --node_budget_per_type 20000 \
      --epochs 50 --defense_epochs 30 \
      --defenses hr,unlearn \
      --save_results --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch
from torch_geometric.data import HeteroData

from data.hetero_dataset import get_dataset
from attack.hetero_attack import create_attacker
from defense.hetero_guard import HeteroGuard
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


def cuda_reset_peak(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_mb(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return float(max(allocated, reserved))


def count_hetero_graph(data: HeteroData) -> Dict:
    node_counts = {}
    edge_counts = {}
    total_nodes = 0
    total_edges = 0

    for ntype in data.node_types:
        if "x" in data[ntype] and data[ntype].x is not None:
            n = int(data[ntype].x.size(0))
        elif getattr(data[ntype], "num_nodes", None) is not None:
            n = int(data[ntype].num_nodes)
        else:
            n = 0
        node_counts[ntype] = n
        total_nodes += n

    for etype in data.edge_types:
        if "edge_index" in data[etype] and data[etype].edge_index is not None:
            e = int(data[etype].edge_index.size(1))
        else:
            e = 0
        edge_counts[str(etype)] = e
        total_edges += e

    return {
        "num_node_types": len(data.node_types),
        "num_edge_types": len(data.edge_types),
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
    }


def ensure_masks(data: HeteroData, target_type: str, seed: int, train_ratio=0.6, val_ratio=0.2):
    """
    Ensure target node train/val/test masks exist.
    If labels exist but masks are missing, create a deterministic split.
    """
    store = data[target_type]
    if "train_mask" in store and "val_mask" in store and "test_mask" in store:
        return data

    if "y" not in store:
        raise RuntimeError(f"Target node type {target_type} has no y labels; cannot create split.")

    n = int(store.y.size(0))
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    store.train_mask = train_mask
    store.val_mask = val_mask
    store.test_mask = test_mask
    return data


def load_ogbn_mag_fallback(data_dir: str, seed: int):
    """
    Fallback loader for OGBN-MAG using PyG's OGB_MAG dataset.
    Requires:
        pip install ogb torch-geometric
    """
    try:
        from torch_geometric.datasets import OGB_MAG
        import torch_geometric.transforms as T
    except Exception as e:
        raise RuntimeError(
            "get_dataset('OGBN-MAG') failed and PyG OGB_MAG fallback is unavailable. "
            "Install ogb and torch-geometric, or add OGBN-MAG support to data/hetero_dataset.py."
        ) from e

    root = os.path.join(data_dir, "OGBN-MAG")
    dataset = OGB_MAG(root=root, preprocess="metapath2vec", transform=T.ToUndirected())
    data = dataset[0]
    target_type = "paper"
    num_classes = int(dataset.num_classes)

    # PyG OGB_MAG usually has masks. Create deterministic split if absent.
    data = ensure_masks(data, target_type=target_type, seed=seed)
    return data, num_classes, target_type


def load_scalability_dataset(dataset_name: str, data_dir: str, seed: int):
    name = dataset_name.lower().replace("_", "-")
    if name in {"ogbn-mag", "ogb-mag", "mag"}:
        try:
            return get_dataset("OGBN-MAG", data_dir=data_dir, validate=True, verbose=False)
        except Exception:
            return load_ogbn_mag_fallback(data_dir=data_dir, seed=seed)
    return get_dataset(dataset_name, data_dir=data_dir, validate=True, verbose=False)


def subset_hetero_graph(
    data: HeteroData,
    target_type: str,
    node_budget_per_type: int,
    seed: int,
) -> HeteroData:
    """
    Sample an induced hetero subgraph with up to node_budget_per_type nodes per type.
    This is deterministic and preserves target train/val/test masks.

    If node_budget_per_type <= 0, returns data unchanged.
    """
    if node_budget_per_type is None or node_budget_per_type <= 0:
        return data

    g = torch.Generator()
    g.manual_seed(seed)

    selected = {}
    old_to_new = {}

    for ntype in data.node_types:
        store = data[ntype]
        if "x" in store and store.x is not None:
            n = int(store.x.size(0))
        elif getattr(store, "num_nodes", None) is not None:
            n = int(store.num_nodes)
        else:
            continue

        budget = min(int(node_budget_per_type), n)

        # For target nodes, keep a balanced portion from train/val/test if masks exist.
        if ntype == target_type and all(k in store for k in ["train_mask", "val_mask", "test_mask"]):
            idx_parts = []
            for mask_name in ["train_mask", "val_mask", "test_mask"]:
                idx = torch.nonzero(store[mask_name].detach().cpu(), as_tuple=False).view(-1)
                if idx.numel() > 0:
                    part_budget = max(1, budget // 3)
                    perm = idx[torch.randperm(idx.numel(), generator=g)[:min(part_budget, idx.numel())]]
                    idx_parts.append(perm)
            idx = torch.unique(torch.cat(idx_parts)) if idx_parts else torch.empty(0, dtype=torch.long)

            if idx.numel() < budget:
                remaining = torch.ones(n, dtype=torch.bool)
                remaining[idx] = False
                rem_idx = torch.nonzero(remaining, as_tuple=False).view(-1)
                add = rem_idx[torch.randperm(rem_idx.numel(), generator=g)[:min(budget - idx.numel(), rem_idx.numel())]]
                idx = torch.unique(torch.cat([idx, add]))
        else:
            idx = torch.randperm(n, generator=g)[:budget]

        idx = idx.sort()[0].long()
        selected[ntype] = idx
        mapping = torch.full((n,), -1, dtype=torch.long)
        mapping[idx] = torch.arange(idx.numel(), dtype=torch.long)
        old_to_new[ntype] = mapping

    out = HeteroData()

    for ntype, idx in selected.items():
        store = data[ntype]
        for key, value in store.items():
            if isinstance(value, torch.Tensor):
                if value.size(0) == old_to_new[ntype].numel():
                    out[ntype][key] = value[idx].clone()
                else:
                    out[ntype][key] = value.clone()
            else:
                out[ntype][key] = deepcopy(value)
        if "x" in out[ntype] and out[ntype].x is not None:
            out[ntype].num_nodes = int(out[ntype].x.size(0))
        else:
            out[ntype].num_nodes = int(idx.numel())

    for etype in data.edge_types:
        src_type, rel, dst_type = etype
        if src_type not in selected or dst_type not in selected:
            continue
        if "edge_index" not in data[etype]:
            continue

        ei = data[etype].edge_index.detach().cpu()
        src_map = old_to_new[src_type]
        dst_map = old_to_new[dst_type]
        src_new = src_map[ei[0]]
        dst_new = dst_map[ei[1]]
        keep = (src_new >= 0) & (dst_new >= 0)

        new_ei = torch.stack([src_new[keep], dst_new[keep]], dim=0).long()
        out[etype].edge_index = new_ei

        for key, value in data[etype].items():
            if key == "edge_index":
                continue
            if isinstance(value, torch.Tensor) and value.size(0) == ei.size(1):
                out[etype][key] = value.detach().cpu()[keep].clone()
            elif key != "edge_index":
                out[etype][key] = value.clone() if isinstance(value, torch.Tensor) else deepcopy(value)

    out = ensure_masks(out, target_type=target_type, seed=seed)
    return out


def eval_model(model, clean_data, poison_data, eval_graph, target_type, target_class, attacker, num_inject, device, seed):
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
    return {
        "defense_clean_acc": clean_acc,
        "defense_asr": 0.0 if asr is None else float(asr),
        "defense_asr_clean_acc": 0.0 if asr_clean_acc is None else float(asr_clean_acc),
        "natural_target_ratio": 0.0 if natural_ratio is None else float(natural_ratio),
    }


def run_one_defense(
    args,
    defense_name: str,
    clean_data,
    poison_data,
    target_type,
    num_classes,
    attacker,
    true_poison,
    attack_metrics,
    graph_stats,
    device,
    seed_offset: int,
):
    if defense_name not in {"hr", "unlearn"}:
        raise ValueError(f"Unsupported scalability defense: {defense_name}")

    cuda_reset_peak(device)
    t0 = time.time()

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

    pretrain_t0 = time.time()
    pre_info = guard.pretrain_reference(
        epochs=args.pretrain_epochs,
        lr=args.pretrain_lr,
        weight_decay=args.defense_weight_decay,
        verbose=args.verbose,
    )
    pretrain_time = time.time() - pretrain_t0

    detect_t0 = time.time()
    suspicious, scores, det_metrics = guard.detect(
        target_class=attacker.target_class,
        top_k_ratio=args.detection_ratio,
        true_poison_indices=true_poison,
        return_metrics=True,
        use_label_signal=args.use_label_signal,
    )
    detection_time = time.time() - detect_t0

    train_t0 = time.time()
    train_info = guard.train_defense(
        epochs=args.defense_epochs,
        lr=args.defense_lr,
        weight_decay=args.defense_weight_decay,
        verbose=args.verbose,
        use_clean_graph=False,
        use_prune=True,
        min_weight=args.min_weight,
        max_downweight=args.max_downweight,
        hard_remove_suspicious=True,
        use_trigger_unlearning=(defense_name == "unlearn"),
        attacker=attacker if defense_name == "unlearn" else None,
        target_class=attacker.target_class,
        unlearn_lambda=args.unlearn_lambda,
        unlearn_samples=args.unlearn_samples,
        target_suppression=args.target_suppression,
        unlearn_exclude_target=not args.include_target_in_unlearning,
    )
    train_time = time.time() - train_t0

    model = guard.get_model()
    eval_graph = guard.purified_data if guard.purified_data is not None else poison_data

    eval_t0 = time.time()
    metrics = eval_model(
        model=model,
        clean_data=clean_data,
        poison_data=poison_data,
        eval_graph=eval_graph,
        target_type=target_type,
        target_class=attacker.target_class,
        attacker=attacker,
        num_inject=args.num_inject,
        device=device,
        seed=args.seed,
    )
    eval_time = time.time() - eval_t0

    total_time = time.time() - t0
    peak_memory = cuda_peak_mb(device)

    metrics.update({
        "defense_name": "HeteroGuard-HR" if defense_name == "hr" else "HeteroGuard-Unlearn",
        "defense_key": defense_name,
        "detection_precision": float(det_metrics.get("precision", 0.0)),
        "detection_recall": float(det_metrics.get("recall", 0.0)),
        "detection_f1": float(det_metrics.get("f1", 0.0)),
        "suspicious_nodes": int(suspicious.numel()),
        "hard_removed_train_nodes": int(train_info.get("hard_removed_train_nodes", 0)),
        "asr_drop": float(attack_metrics["poisoned_asr"] - metrics["defense_asr"]),
        "utility_drop": float(attack_metrics["clean_acc"] - metrics["defense_clean_acc"]),
        "relative_asr_reduction": float(
            (attack_metrics["poisoned_asr"] - metrics["defense_asr"]) /
            max(attack_metrics["poisoned_asr"], 1e-12)
        ),
        "defense_time_sec": float(total_time),
        "pretrain_time_sec": float(pretrain_time),
        "detection_time_sec": float(detection_time),
        "defense_train_time_sec": float(train_time),
        "eval_time_sec": float(eval_time),
        "peak_gpu_memory_mb": float(peak_memory),
        "processed_total_nodes": int(graph_stats["total_nodes"]),
        "processed_total_edges": int(graph_stats["total_edges"]),
        "processed_num_node_types": int(graph_stats["num_node_types"]),
        "processed_num_edge_types": int(graph_stats["num_edge_types"]),
        "processed_node_counts": graph_stats["node_counts"],
        "processed_edge_counts": graph_stats["edge_counts"],
        "pretrain_info": pre_info,
        "train_info": train_info,
    })
    return metrics


def run_scalability(args):
    configure_reproducibility(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    total_t0 = time.time()
    if device.type == "cuda":
        cuda_reset_peak(device)

    raw_data, num_classes, target_type = load_scalability_dataset(args.dataset, args.data_dir, args.seed)

    if args.node_budget_per_type > 0:
        raw_stats = count_hetero_graph(raw_data)
        data = subset_hetero_graph(
            raw_data,
            target_type=target_type,
            node_budget_per_type=args.node_budget_per_type,
            seed=args.seed,
        )
        sample_mode = "budgeted_induced_subgraph"
    else:
        raw_stats = count_hetero_graph(raw_data)
        data = raw_data
        sample_mode = "full_graph"

    data = data.to(device)
    validate_edge_index(data, raise_error=True)
    graph_stats = count_hetero_graph(data)

    # Clean model.
    configure_reproducibility(args.seed + 101)
    clean_t0 = time.time()
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
    clean_train_time = time.time() - clean_t0
    clean_acc = float(measure_clean_acc(clean_model, data, target_type, device=device))

    # Attack.
    configure_reproducibility(args.seed + 202)
    attack_t0 = time.time()
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
    attack_generation_time = time.time() - attack_t0

    true_poison = attacker.get_poisoned_nodes()

    # Poisoned model.
    configure_reproducibility(args.seed + 303)
    poison_t0 = time.time()
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
    poison_train_time = time.time() - poison_t0
    backdoored_clean_acc = float(measure_clean_acc(poisoned_model, poison_data, target_type, device=device))

    poisoned_asr, poisoned_asr_clean_acc, natural_ratio = measure_asr(
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
    poisoned_asr = 0.0 if poisoned_asr is None else float(poisoned_asr)
    poisoned_asr_clean_acc = 0.0 if poisoned_asr_clean_acc is None else float(poisoned_asr_clean_acc)
    natural_ratio = 0.0 if natural_ratio is None else float(natural_ratio)

    attack_metrics = {
        "clean_acc": clean_acc,
        "backdoored_clean_acc": backdoored_clean_acc,
        "clean_acc_drop": clean_acc - backdoored_clean_acc,
        "poisoned_asr": poisoned_asr,
        "poisoned_asr_clean_acc": poisoned_asr_clean_acc,
        "natural_target_ratio": natural_ratio,
        "n_poison_nodes": int(true_poison.numel()),
    }

    selected = [x.strip().lower() for x in args.defenses.split(",") if x.strip()]
    if "all" in selected:
        selected = ["hr", "unlearn"]

    defenses = {}
    for i, dname in enumerate(selected):
        print("=" * 80)
        print(f"[SCALABILITY] defense={dname} dataset={args.dataset} model={args.model} attack={args.attack} seed={args.seed}")
        print("=" * 80)
        defenses[dname] = run_one_defense(
            args=args,
            defense_name=dname,
            clean_data=data,
            poison_data=poison_data,
            target_type=target_type,
            num_classes=num_classes,
            attacker=attacker,
            true_poison=true_poison,
            attack_metrics=attack_metrics,
            graph_stats=graph_stats,
            device=device,
            seed_offset=500 + i * 100,
        )

    total_runtime = time.time() - total_t0
    peak_total_gpu_mb = cuda_peak_mb(device)

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
        "sample_mode": sample_mode,
        "node_budget_per_type": int(args.node_budget_per_type),
        "raw_graph_stats": raw_stats,
        "processed_graph_stats": graph_stats,
        "attack_metrics": attack_metrics,
        "defenses": defenses,
        "runtime": {
            "clean_train_time_sec": float(clean_train_time),
            "attack_generation_time_sec": float(attack_generation_time),
            "poison_train_time_sec": float(poison_train_time),
            "total_runtime_sec": float(total_runtime),
            "peak_total_gpu_memory_mb": float(peak_total_gpu_mb),
        },
        "clean_train_info": clean_info,
        "poison_train_info": poison_info,
        "attacker_info": attacker.get_attack_info() if hasattr(attacker, "get_attack_info") else {},
        "config": vars(args),
    }

    if args.save_results:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / (
            f"scalability_{args.dataset}_{args.model}_{args.attack}_"
            f"budget{args.node_budget_per_type}_r{args.poison_rate:.3f}_seed{args.seed}_{ts}.json"
        )
        path.write_text(json.dumps(_jsonable(result), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SAVED] {path}")

    print("=" * 80)
    print(f"Scalability summary | {args.dataset} {args.model} {args.attack} seed={args.seed}")
    print("=" * 80)
    print(
        f"sample_mode={sample_mode}, nodes={graph_stats['total_nodes']}, edges={graph_stats['total_edges']}, "
        f"Clean={clean_acc:.4f}, BA={backdoored_clean_acc:.4f}, PoisonASR={poisoned_asr:.4f}"
    )
    for key, d in defenses.items():
        print(
            f"{d['defense_name']:22s} | ASR={d['defense_asr']:.4f}, "
            f"Clean={d['defense_clean_acc']:.4f}, Time={d['defense_time_sec']:.2f}s, "
            f"PeakGPU={d['peak_gpu_memory_mb']:.1f}MB"
        )
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Run OGBN-MAG scalability study.")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results_scalability_ogbn_mag")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--dataset", type=str, default="OGBN-MAG")
    parser.add_argument("--model", type=str, default="HeteroSAGE")
    parser.add_argument("--attack", type=str, default="relation")
    parser.add_argument("--defenses", type=str, default="hr,unlearn")

    parser.add_argument("--node_budget_per_type", type=int, default=50000,
                        help="0 means full graph; >0 samples up to this many nodes per type.")

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_bases", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--clean_epochs", type=int, default=None)
    parser.add_argument("--poison_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--poison_save_best", type=str, default="last", choices=["last", "train_loss", "val_acc"])

    parser.add_argument("--poison_rate", type=float, default=0.05)
    parser.add_argument("--trigger_size", type=int, default=5)
    parser.add_argument("--target_class", type=int, default=0)
    parser.add_argument("--trigger_strength", type=float, default=3.0)
    parser.add_argument("--surrogate_epochs", type=int, default=20)
    parser.add_argument("--num_inject", type=int, default=500)

    parser.add_argument("--relation_mode", type=str, default="hybrid", choices=["pure", "hybrid"])
    parser.add_argument("--target_feature_strength", type=float, default=3.0)
    parser.add_argument("--aux_feature_strength", type=float, default=4.0)
    parser.add_argument("--no_aux_clique", action="store_true")

    parser.add_argument("--pretrain_epochs", type=int, default=30)
    parser.add_argument("--pretrain_lr", type=float, default=0.003)
    parser.add_argument("--defense_epochs", type=int, default=50)
    parser.add_argument("--defense_lr", type=float, default=0.003)
    parser.add_argument("--defense_weight_decay", type=float, default=1e-4)
    parser.add_argument("--detection_ratio", type=float, default=0.05)
    parser.add_argument("--use_label_signal", action="store_true")
    parser.add_argument("--min_weight", type=float, default=0.0)
    parser.add_argument("--max_downweight", type=float, default=1.0)

    parser.add_argument("--unlearn_lambda", type=float, default=1.0)
    parser.add_argument("--unlearn_samples", type=int, default=512)
    parser.add_argument("--target_suppression", type=float, default=0.2)
    parser.add_argument("--include_target_in_unlearning", action="store_true")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_scalability(args)
