# -*- coding: utf-8 -*-
"""
Experiment 6: Robustness runner for HeteroGuard-Unlearn.

Supports:
1. inaccurate_detection_ratio
2. partial_detection_failure
3. adaptive_attack

The runner evaluates only HeteroGuard-Unlearn and saves a tagged JSON file.

Example:
    python -m experiments.run_robustness_suite \
      --dataset ACM --model HAN --attack relation \
      --poison_rate 0.2 --trigger_size 10 --detection_ratio 0.1 \
      --robustness_group inaccurate_detection_ratio \
      --robustness_value 0.5 \
      --epochs 200 --save_results --seed 1
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch

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


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _align_clean_metadata(clean_data, poison_data, device):
    try:
        from defense.utils import align_clean_to_poison_metadata
        return align_clean_to_poison_metadata(clean_data, poison_data).to(device)
    except Exception:
        return poison_data


def _adjust_scores_after_dropping(scores, suspicious, kept, dropped, num_nodes, device):
    if scores is None:
        return None

    scores = scores.detach().clone().float().to(device)

    # Case 1: scores is full target-node-length vector.
    if scores.numel() == num_nodes:
        if dropped.numel() > 0:
            scores[dropped.to(device)] = 0.0
        return scores

    # Case 2: scores only correspond to suspicious nodes.
    if scores.numel() == suspicious.numel():
        full = torch.zeros(num_nodes, dtype=torch.float, device=device)
        suspicious_dev = suspicious.to(device)
        full[suspicious_dev] = scores
        if dropped.numel() > 0:
            full[dropped.to(device)] = 0.0
        return full

    # Fallback: do not break training.
    return scores


def apply_partial_detection_failure(guard, target_type, drop_ratio: float, seed: int, device):
    """
    Randomly remove a fraction of detected suspicious target nodes before defense training.
    This simulates imperfect recall.
    """
    suspicious = guard.suspicious_nodes
    scores = guard.suspicious_scores

    if suspicious is None or suspicious.numel() == 0 or drop_ratio <= 0:
        return {
            "drop_ratio": float(drop_ratio),
            "original_suspicious": int(0 if suspicious is None else suspicious.numel()),
            "kept_suspicious": int(0 if suspicious is None else suspicious.numel()),
            "dropped_suspicious": 0,
        }

    suspicious = suspicious.detach().cpu().long()
    n = suspicious.numel()
    n_drop = int(round(float(drop_ratio) * n))
    n_drop = max(0, min(n_drop, n))

    g = torch.Generator()
    g.manual_seed(int(seed) + 777)
    perm = torch.randperm(n, generator=g)
    drop_local = perm[:n_drop]
    keep_local = perm[n_drop:]

    dropped = suspicious[drop_local]
    kept = suspicious[keep_local]

    num_nodes = int(guard.poison_data[target_type].x.size(0))
    guard.suspicious_nodes = kept.to(device)
    guard.suspicious_scores = _adjust_scores_after_dropping(
        scores=scores,
        suspicious=suspicious,
        kept=kept,
        dropped=dropped,
        num_nodes=num_nodes,
        device=device,
    )

    return {
        "drop_ratio": float(drop_ratio),
        "original_suspicious": int(n),
        "kept_suspicious": int(kept.numel()),
        "dropped_suspicious": int(dropped.numel()),
    }


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


def run_robustness(args):
    configure_reproducibility(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    t0 = time.time()

    data, num_classes, target_type = get_dataset(args.dataset, data_dir=args.data_dir, validate=True, verbose=False)
    data = data.to(device)
    validate_edge_index(data, raise_error=True)

    # Clean model.
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

    # Poison graph.
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

    # Poisoned model.
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

    # Clean-trigger diagnostic.
    configure_reproducibility(args.seed + 404)
    clean_probe_train_data = _align_clean_metadata(data, poison_data, device)
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
    clean_trigger_asr, clean_trigger_clean_acc, _ = measure_asr(
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
        "poisoned_asr": poisoned_asr,
        "poisoned_asr_clean_acc": poisoned_asr_clean_acc,
        "natural_target_ratio": natural_ratio,
        "learned_backdoor_gap": poisoned_asr - clean_trigger_asr,
        "n_poison_nodes": int(true_poison.numel()),
    }

    # HeteroGuard-Unlearn defense.
    configure_reproducibility(args.seed + 505)
    guard = HeteroGuard(
        clean_data=data,
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
        target_class=attacker.target_class,
        top_k_ratio=args.detection_ratio,
        true_poison_indices=true_poison,
        return_metrics=True,
        use_label_signal=args.use_label_signal,
    )

    partial_info = None
    if args.robustness_group == "partial_detection_failure":
        partial_info = apply_partial_detection_failure(
            guard=guard,
            target_type=target_type,
            drop_ratio=args.drop_detected_ratio,
            seed=args.seed,
            device=device,
        )

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
    defense_metrics = eval_model(
        model=model,
        clean_data=data,
        poison_data=poison_data,
        eval_graph=eval_graph,
        target_type=target_type,
        target_class=attacker.target_class,
        attacker=attacker,
        num_inject=args.num_inject,
        device=device,
        seed=args.seed,
    )

    defense_metrics.update({
        "detection_precision": float(det_metrics.get("precision", 0.0)),
        "detection_recall": float(det_metrics.get("recall", 0.0)),
        "detection_f1": float(det_metrics.get("f1", 0.0)),
        "suspicious_nodes": int(suspicious.numel()),
        "hard_removed_train_nodes": int(train_info.get("hard_removed_train_nodes", 0)),
        "asr_drop": poisoned_asr - defense_metrics["defense_asr"],
        "utility_drop": clean_acc - defense_metrics["defense_clean_acc"],
        "relative_asr_reduction": (poisoned_asr - defense_metrics["defense_asr"]) / max(poisoned_asr, 1e-12),
        "partial_detection_failure": partial_info,
        "pretrain_info": pre_info,
        "train_info": train_info,
    })

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
        "robustness_group": args.robustness_group,
        "robustness_value": args.robustness_value,
        "drop_detected_ratio": float(args.drop_detected_ratio),
        "adaptive_variant": args.adaptive_variant,
        "attack_metrics": attack_metrics,
        "defense": defense_metrics,
        "attacker_info": attacker.get_attack_info() if hasattr(attacker, "get_attack_info") else {},
        "config": vars(args),
        "runtime_sec": time.time() - t0,
    }

    if args.save_results:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_group = str(args.robustness_group).replace("/", "_")
        safe_value = str(args.robustness_value).replace("/", "_")
        path = out_dir / (
            f"robustness_{safe_group}_{safe_value}_{args.dataset}_{args.model}_"
            f"{args.attack}_r{args.poison_rate:.3f}_seed{args.seed}_{ts}.json"
        )
        path.write_text(json.dumps(_jsonable(result), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SAVED] {path}")

    print("=" * 80)
    print(
        f"Robustness summary | group={args.robustness_group} value={args.robustness_value} "
        f"{args.dataset} {args.model} {args.attack} seed={args.seed}"
    )
    print("=" * 80)
    print(
        f"Clean={clean_acc:.4f}, BA={backdoored_clean_acc:.4f}, "
        f"PoisonASR={poisoned_asr:.4f}, CleanTrigASR={clean_trigger_asr:.4f}, "
        f"DefenseASR={defense_metrics['defense_asr']:.4f}, "
        f"DefenseAcc={defense_metrics['defense_clean_acc']:.4f}, "
        f"DetF1={defense_metrics['detection_f1']:.4f}"
    )
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Run robustness experiment for HeteroGuard-Unlearn.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results_robustness_acm_han")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--dataset", type=str, default="ACM")
    parser.add_argument("--model", type=str, default="HAN")
    parser.add_argument("--attack", type=str, default="relation")

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

    parser.add_argument("--robustness_group", type=str, required=True,
                        choices=["inaccurate_detection_ratio", "partial_detection_failure", "adaptive_attack"])
    parser.add_argument("--robustness_value", type=str, required=True)
    parser.add_argument("--drop_detected_ratio", type=float, default=0.0)
    parser.add_argument("--adaptive_variant", type=str, default=None)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_robustness(args)
