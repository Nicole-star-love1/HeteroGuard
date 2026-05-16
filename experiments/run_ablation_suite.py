# -*- coding: utf-8 -*-
"""
Experiment 4: HeteroGuard-Unlearn ablation suite.

Setting:
    Dataset: ACM
    Model: HAN
    Attacks: uba, relation, clean_label, cba
    Seeds: 1..5

Ablation variants:
    full
    wo_structural_signal
    wo_feature_signal
    wo_embedding_signal
    wo_hard_removal
    wo_trigger_purification
    wo_trigger_unlearning
    wo_target_suppression
    only_hard_removal
    only_unlearning

Example:
    python -m experiments.run_ablation_suite \
      --dataset ACM --model HAN --attack uba \
      --poison_rate 0.2 --trigger_size 10 \
      --epochs 200 --seed 1 --save_results

Notes:
    The signal-level ablations rely on detector signal names. The script uses
    common aliases. If wo_structural/feature/embedding results equal full exactly,
    inspect detect_info in the output JSON and adjust DISABLE_SIGNAL_ALIASES.
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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


VARIANT_ORDER = [
    "full",
    "wo_structural_signal",
    "wo_feature_signal",
    "wo_embedding_signal",
    "wo_hard_removal",
    "wo_trigger_purification",
    "wo_trigger_unlearning",
    "wo_target_suppression",
    "only_hard_removal",
    "only_unlearning",
]

# These aliases are deliberately broad because detector implementations often
# use different names. If the detector ignores unknown names, they are harmless.
# If your detector has exact signal names, refine these after inspecting detect_info.
DISABLE_SIGNAL_ALIASES = {
    "wo_structural_signal": [
        "structural", "structure", "relation", "degree", "edge", "topology", "meta_path", "metapath"
    ],
    "wo_feature_signal": [
        "feature", "features", "x", "feature_anomaly", "feature_shift", "attribute"
    ],
    "wo_embedding_signal": [
        "embedding", "emb", "representation", "prediction", "confidence", "logit", "posterior"
    ],
}


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


def build_guard(args, clean_data, poison_data, target_type, num_classes, device, seed_offset: int):
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
    return guard, pre_info


def get_variant_config(variant: str, args) -> Dict:
    cfg = {
        "variant": variant,
        "use_prune": True,
        "hard_remove_suspicious": True,
        "use_trigger_unlearning": True,
        "target_suppression": args.target_suppression,
        "unlearn_lambda": args.unlearn_lambda,
        "disable_signals": None,
        "description": "Full HeteroGuard-Unlearn",
    }

    if variant == "full":
        pass

    elif variant == "wo_structural_signal":
        cfg["disable_signals"] = DISABLE_SIGNAL_ALIASES["wo_structural_signal"]
        cfg["description"] = "Full model without structural detection signals"

    elif variant == "wo_feature_signal":
        cfg["disable_signals"] = DISABLE_SIGNAL_ALIASES["wo_feature_signal"]
        cfg["description"] = "Full model without feature detection signals"

    elif variant == "wo_embedding_signal":
        cfg["disable_signals"] = DISABLE_SIGNAL_ALIASES["wo_embedding_signal"]
        cfg["description"] = "Full model without embedding/prediction detection signals"

    elif variant == "wo_hard_removal":
        cfg["hard_remove_suspicious"] = False
        cfg["description"] = "Full model without hard removal of suspicious training nodes"

    elif variant == "wo_trigger_purification":
        cfg["use_prune"] = False
        cfg["description"] = "Full model without trigger purification / edge pruning"

    elif variant == "wo_trigger_unlearning":
        cfg["use_trigger_unlearning"] = False
        cfg["target_suppression"] = 0.0
        cfg["unlearn_lambda"] = 0.0
        cfg["description"] = "HeteroGuard-HR without trigger unlearning"

    elif variant == "wo_target_suppression":
        cfg["target_suppression"] = 0.0
        cfg["description"] = "Full model without target-class probability suppression"

    elif variant == "only_hard_removal":
        cfg["use_prune"] = False
        cfg["hard_remove_suspicious"] = True
        cfg["use_trigger_unlearning"] = False
        cfg["target_suppression"] = 0.0
        cfg["unlearn_lambda"] = 0.0
        cfg["description"] = "Only hard-remove detected suspicious training nodes"

    elif variant == "only_unlearning":
        cfg["use_prune"] = False
        cfg["hard_remove_suspicious"] = False
        cfg["use_trigger_unlearning"] = True
        cfg["description"] = "Only trigger unlearning without purification or hard removal"

    else:
        raise ValueError(f"Unknown ablation variant: {variant}")

    return cfg


def run_one_variant(
    args,
    variant: str,
    clean_data,
    poison_data,
    target_type,
    num_classes,
    true_poison,
    attacker,
    attack_metrics,
    device,
    seed_offset: int,
) -> Dict:
    cfg = get_variant_config(variant, args)
    t0 = time.time()

    guard, pre_info = build_guard(
        args=args,
        clean_data=clean_data,
        poison_data=poison_data,
        target_type=target_type,
        num_classes=num_classes,
        device=device,
        seed_offset=seed_offset,
    )

    suspicious, scores, det_metrics = guard.detect(
        target_class=attacker.target_class,
        top_k_ratio=args.detection_ratio,
        true_poison_indices=true_poison,
        return_metrics=True,
        disable_signals=cfg["disable_signals"],
        use_label_signal=args.use_label_signal,
    )

    train_info = guard.train_defense(
        epochs=args.defense_epochs,
        lr=args.defense_lr,
        weight_decay=args.defense_weight_decay,
        verbose=args.verbose,
        use_clean_graph=False,
        use_prune=cfg["use_prune"],
        min_weight=args.min_weight,
        max_downweight=args.max_downweight,
        hard_remove_suspicious=cfg["hard_remove_suspicious"],
        use_trigger_unlearning=cfg["use_trigger_unlearning"],
        attacker=attacker,
        target_class=attacker.target_class,
        unlearn_lambda=cfg["unlearn_lambda"],
        unlearn_samples=args.unlearn_samples,
        target_suppression=cfg["target_suppression"],
        unlearn_exclude_target=not args.include_target_in_unlearning,
    )

    model = guard.get_model()
    eval_graph = guard.purified_data if guard.purified_data is not None else poison_data

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

    metrics.update({
        "variant": variant,
        "variant_description": cfg["description"],
        "use_prune": bool(cfg["use_prune"]),
        "hard_remove_suspicious": bool(cfg["hard_remove_suspicious"]),
        "use_trigger_unlearning": bool(cfg["use_trigger_unlearning"]),
        "target_suppression": float(cfg["target_suppression"]),
        "unlearn_lambda": float(cfg["unlearn_lambda"]),
        "disable_signals": cfg["disable_signals"],
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
        "pretrain_info": pre_info,
        "train_info": train_info,
        "detect_info": getattr(guard, "detect_info", {}),
        "runtime_sec": time.time() - t0,
    })
    return metrics


def run_ablation_suite(args):
    configure_reproducibility(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    total_t0 = time.time()

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

    # 2. Poison graph.
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
    # Use metadata-aligned clean graph when poison graph has extra edge types.
    configure_reproducibility(args.seed + 404)
    try:
        from defense.utils import align_clean_to_poison_metadata
        clean_probe_train_data = align_clean_to_poison_metadata(data, poison_data).to(device)
    except Exception:
        clean_probe_train_data = poison_data

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

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    if "all" in variants:
        variants = VARIANT_ORDER

    ablations = {}
    for i, variant in enumerate(variants):
        print("=" * 80)
        print(f"[ABLATION] {args.dataset} {args.model} {args.attack} seed={args.seed} variant={variant}")
        print("=" * 80)
        ablations[variant] = run_one_variant(
            args=args,
            variant=variant,
            clean_data=data,
            poison_data=poison_data,
            target_type=target_type,
            num_classes=num_classes,
            true_poison=true_poison,
            attacker=attacker,
            attack_metrics=attack_metrics,
            device=device,
            seed_offset=600 + i * 100,
        )

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
        "ablations": ablations,
        "attacker_info": attacker.get_attack_info() if hasattr(attacker, "get_attack_info") else {},
        "config": vars(args),
        "runtime_sec": time.time() - total_t0,
    }

    if args.save_results:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"ablation_{args.dataset}_{args.model}_{args.attack}_r{args.poison_rate:.3f}_seed{args.seed}_{ts}.json"
        path.write_text(json.dumps(_jsonable(result), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SAVED] {path}")

    print("=" * 80)
    print(f"Ablation summary | {args.dataset} {args.model} {args.attack} seed={args.seed}")
    print("=" * 80)
    print(
        f"Clean={clean_acc:.4f}, BA={backdoored_clean_acc:.4f}, "
        f"PoisonASR={poison_asr:.4f}, CleanTrigASR={clean_trigger_asr:.4f}, "
        f"Gap={poison_asr-clean_trigger_asr:+.4f}"
    )
    for variant, v in ablations.items():
        print(
            f"{variant:24s} | Clean={v['defense_clean_acc']:.4f}, "
            f"ASR={v['defense_asr']:.4f}, DetF1={v['detection_f1']:.4f}, "
            f"Drop={v['asr_drop']:+.4f}"
        )
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Run HeteroGuard-Unlearn ablation suite.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results_ablation_acm_han")
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

    parser.add_argument("--variants", type=str, default="all",
                        help="Comma list of variants or all.")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_ablation_suite(args)
