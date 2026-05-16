# -*- coding: utf-8 -*-
"""
Integrated in-memory experiment runner.

Purpose:
    Avoid Step1/Step2 split, avoid model checkpoint save/load, and avoid
    re-generating poison graphs in the defense stage.

This script runs everything in one process:
    1. load dataset
    2. train clean model
    3. create attacker and poison graph
    4. train poisoned model in memory
    5. evaluate poisoned ASR
    6. train clean-trigger probe after poisoned training
    7. optionally run HeteroGuard defense in memory

Key design choices:
    - No model checkpoint is saved by default.
    - No Step2 re-poisoning is performed.
    - Poisoned model uses --poison_save_best last by default.
    - Clean-trigger diagnostic is run after poisoned training so it cannot
      affect poisoned-model initialization/dropout RNG.
    - Results can optionally be saved as JSON by --save_results.

Examples:
    python -m experiments.run_integrated \
      --dataset ACM --model HAN --attack relation \
      --poison_rate 0.2 --trigger_size 10 --epochs 200 --debug

    python -m experiments.run_integrated \
      --dataset ACM --model HGT --attack relation \
      --poison_rate 0.2 --trigger_size 10 --epochs 200 --debug

    python -m experiments.run_integrated \
      --dataset ACM --model HAN --attack uba \
      --poison_rate 0.2 --trigger_size 10 --epochs 200 --debug

    # relation-pure:
    python -m experiments.run_integrated \
      --dataset ACM --model HAN --attack relation \
      --poison_rate 0.2 --trigger_size 10 --relation_mode pure --epochs 200

    # relation-hybrid with stronger target feature boost:
    python -m experiments.run_integrated \
      --dataset ACM --model HAN --attack relation \
      --poison_rate 0.2 --trigger_size 10 --relation_mode hybrid \
      --target_feature_strength 6.0 --aux_feature_strength 8.0 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from copy import deepcopy
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.hetero_dataset import get_dataset
from models.hetero_gnn import create_model
from attack.hetero_attack import create_attacker
from experiments.utils import (
    clone_hetero_data,
    measure_asr,
    measure_clean_acc,
    set_seed,
    validate_edge_index,
    _jsonable,
)

try:
    from defense.hetero_guard import HeteroGuard
except Exception:
    HeteroGuard = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DATASETS = ["DBLP", "ACM", "IMDB", "Freebase"]
MODELS = ["HAN", "HGT", "RGCN", "HeteroSAGE"]
ATTACKS = ["feature", "sba", "uba", "relation", "clean_label", "cba", "grad"]


# ---------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------

def snapshot_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["torch_cuda"] = None
    return state


def restore_rng_state(state):
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception:
            pass


def configure_reproducibility(seed: int):
    # Best-effort deterministic setup for PyTorch/PyG experiments.
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_model(
    model,
    data,
    target_type: str,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
    save_best: str = "val_acc",
    verbose: bool = False,
):
    """
    Local training loop.

    save_best:
        val_acc     clean model default
        train_loss  optional poisoned model policy
        last        poisoned model default
    """
    if save_best not in {"val_acc", "train_loss", "last"}:
        raise ValueError(f"Unsupported save_best={save_best}")

    validate_edge_index(data, raise_error=True)

    # Lazy build before optimizer.
    model.eval()
    with torch.no_grad():
        _ = model(data)

    first_x = next(iter(data.x_dict.values()))
    model.to(first_x.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mask = data[target_type].train_mask
    val_mask = data[target_type].val_mask
    labels = data[target_type].y

    best_val_acc = -1.0
    best_train_loss = float("inf")
    best_state = None
    best_epoch = 0
    last_loss = None
    last_val_acc = None

    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()

        logits = model(data)[target_type]
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)[target_type]
            pred = logits.argmax(dim=1)
            if val_mask.sum().item() > 0:
                val_acc = (pred[val_mask] == labels[val_mask]).float().mean().item()
            else:
                val_acc = 0.0

        train_loss = float(loss.item())
        last_loss = train_loss
        last_val_acc = float(val_acc)

        should_save = False
        if save_best == "val_acc":
            should_save = val_acc > best_val_acc
        elif save_best == "train_loss":
            should_save = train_loss < best_train_loss
        elif save_best == "last":
            should_save = True

        if should_save:
            if save_best == "val_acc":
                best_val_acc = float(val_acc)
            elif save_best == "train_loss":
                best_train_loss = train_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            logger.info(
                f"    Epoch {epoch + 1}/{epochs}, "
                f"loss={train_loss:.4f}, val_acc={val_acc:.4f}"
            )

        # Do not early-stop poisoned training by clean val_acc.
        if save_best == "val_acc" and patience is not None and patience > 0:
            if (epoch - best_epoch) >= patience:
                if verbose:
                    logger.info(f"    Early stop at epoch {epoch + 1}, best_epoch={best_epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Make sure all modules stay on graph device.
    model.to(first_x.device)

    return {
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "best_train_loss": float(best_train_loss) if best_train_loss < float("inf") else None,
        "last_loss": float(last_loss) if last_loss is not None else None,
        "last_val_acc": float(last_val_acc) if last_val_acc is not None else None,
        "save_best": save_best,
    }


def build_model(args, data, num_classes, target_type, device):
    model = create_model(
        model_name=args.model,
        data=data,
        num_classes=num_classes,
        target_node_type=target_type,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_bases=args.num_bases,
    ).to(device)
    return model


# ---------------------------------------------------------------------
# Attack configuration
# ---------------------------------------------------------------------

def build_attacker_kwargs(args):
    kwargs = {
        "target_class": args.target_class,
        "poison_rate": args.poison_rate,
        "trigger_size": args.trigger_size,
        "seed": args.seed,
        "trigger_strength": args.trigger_strength,
        "surrogate_epochs": args.surrogate_epochs,
    }

    # Relation-specific knobs. Extra kwargs are harmless for current attack classes
    # because BaseHetAttack accepts **kwargs.
    if args.relation_mode == "pure":
        kwargs["target_feature_boost"] = False
    elif args.relation_mode == "hybrid":
        kwargs["target_feature_boost"] = True

    kwargs["target_feature_strength"] = args.target_feature_strength
    kwargs["aux_feature_strength"] = args.aux_feature_strength
    kwargs["use_aux_clique"] = not args.no_aux_clique

    return kwargs


# ---------------------------------------------------------------------
# Integrated run
# ---------------------------------------------------------------------

def run_integrated(args):
    configure_reproducibility(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info("=" * 80)
    logger.info(
        f"Integrated run | dataset={args.dataset}, model={args.model}, "
        f"attack={args.attack}, seed={args.seed}"
    )
    logger.info("=" * 80)

    data, num_classes, target_type = get_dataset(
        args.dataset,
        data_dir=args.data_dir,
        validate=True,
        verbose=False,
    )
    data = data.to(device)
    validate_edge_index(data, raise_error=True)

    logger.info(f"Node types: {data.node_types}")
    logger.info(f"Edge types: {len(data.edge_types)}")
    logger.info(f"Target type: {target_type}, num_classes={num_classes}")
    logger.info(
        f"Split: train={int(data[target_type].train_mask.sum())}, "
        f"val={int(data[target_type].val_mask.sum())}, "
        f"test={int(data[target_type].test_mask.sum())}"
    )

    # 1. Clean model.
    logger.info("Stage 1: Train clean model")
    configure_reproducibility(args.seed + 101)
    clean_model = build_model(args, data, num_classes, target_type, device)
    clean_train_info = train_model(
        clean_model,
        data,
        target_type,
        epochs=args.clean_epochs if args.clean_epochs is not None else args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_best="val_acc",
        verbose=args.verbose,
    )
    clean_acc = float(measure_clean_acc(clean_model, data, target_type, device=device))
    logger.info(f"Clean Acc: {clean_acc:.4f}")

    # 2. Attack and poison graph.
    logger.info("Stage 2: Create attacker and poison graph")
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

    logger.info(f"Poisoned train nodes: {int(true_poison.numel())}")
    logger.info(f"Trigger nodes: {n_trigger}")
    logger.info(f"Target class: {attacker.target_class}")

    # 3. Poisoned model. Run this before clean-trigger probe so diagnostics cannot
    # change poisoned model initialization or dropout trajectories.
    logger.info("Stage 3: Train poisoned model")
    configure_reproducibility(args.seed + 303)
    rng_before_poison_train = snapshot_rng_state()

    poisoned_model = build_model(args, poison_data, num_classes, target_type, device)
    poison_train_info = train_model(
        poisoned_model,
        poison_data,
        target_type,
        epochs=args.poison_epochs if args.poison_epochs is not None else args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_best=args.poison_save_best,
        verbose=args.verbose,
    )

    backdoored_clean_acc = float(
        measure_clean_acc(poisoned_model, poison_data, target_type, device=device)
    )
    logger.info(f"Backdoored Clean Acc: {backdoored_clean_acc:.4f}")

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

    logger.info(f"Poisoned-model ASR: {poison_asr:.4f}")
    logger.info(f"Poisoned-model ASR clean-acc: {poison_asr_clean_acc:.4f}")
    logger.info(f"Natural target ratio: {natural_ratio:.4f}")

    # 4. Clean-trigger diagnostic. Restore RNG first so repeated diagnostics are
    # deterministic and do not contaminate later optional defense.
    logger.info("Stage 4: Clean-model trigger ASR diagnostic")
    restore_rng_state(rng_before_poison_train)
    configure_reproducibility(args.seed + 404)

    clean_probe_model = build_model(args, poison_data, num_classes, target_type, device)

    # Train clean probe on clean labels with poison metadata.
    # We use a clean copy with empty edge stores aligned to poison_data.
    from experiments.utils import align_clean_to_reference_metadata
    clean_probe_data = align_clean_to_reference_metadata(data.cpu(), poison_data.cpu()).to(device)

    clean_probe_info = train_model(
        clean_probe_model,
        clean_probe_data,
        target_type,
        epochs=args.clean_probe_epochs if args.clean_probe_epochs is not None else args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_best="val_acc",
        verbose=args.verbose,
    )
    clean_probe_acc = float(measure_clean_acc(clean_probe_model, clean_probe_data, target_type, device=device))

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
    clean_trigger_natural = 0.0 if clean_trigger_natural is None else float(clean_trigger_natural)

    learned_gap = poison_asr - clean_trigger_asr

    logger.info(f"Clean-trigger probe clean acc: {clean_probe_acc:.4f}")
    logger.info(f"Clean-model Trigger ASR: {clean_trigger_asr:.4f}")
    logger.info(f"Clean-model Trigger clean-acc: {clean_trigger_clean_acc:.4f}")
    logger.info(f"Learned-backdoor gap: {learned_gap:+.4f}")

    result = {
        "dataset": args.dataset,
        "model": args.model,
        "attack": args.attack,
        "seed": args.seed,
        "poison_rate": args.poison_rate,
        "trigger_size": args.trigger_size,
        "relation_mode": args.relation_mode,
        "target_class": int(attacker.target_class),
        "num_classes": int(num_classes),
        "target_type": target_type,
        "metrics": {
            "clean_acc": clean_acc,
            "backdoored_clean_acc": backdoored_clean_acc,
            "poisoned_asr": poison_asr,
            "poisoned_asr_clean_acc": poison_asr_clean_acc,
            "natural_ratio": natural_ratio,
            "clean_probe_acc": clean_probe_acc,
            "clean_trigger_asr": clean_trigger_asr,
            "clean_trigger_clean_acc": clean_trigger_clean_acc,
            "clean_trigger_natural": clean_trigger_natural,
            "learned_backdoor_gap": learned_gap,
            "n_poison_nodes": int(true_poison.numel()),
            "n_trigger_nodes": int(n_trigger),
        },
        "train_info": {
            "clean": clean_train_info,
            "poisoned": poison_train_info,
            "clean_probe": clean_probe_info,
        },
        "attacker_info": attacker.get_attack_info() if hasattr(attacker, "get_attack_info") else {},
        "config": vars(args),
    }

    # 5. Optional in-memory defense.
    if args.run_defense:
        if HeteroGuard is None:
            raise RuntimeError("defense.hetero_guard.HeteroGuard cannot be imported.")

        logger.info("Stage 5: In-memory HeteroGuard defense")
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

        guard.pretrain_reference(
            epochs=args.pretrain_epochs,
            lr=args.pretrain_lr,
            weight_decay=args.defense_weight_decay,
            verbose=args.verbose,
        )

        detection_ratio = args.detection_ratio
        if detection_ratio is None:
            detection_ratio = args.poison_rate

        suspicious, scores, det_metrics = guard.detect(
            target_class=attacker.target_class,
            top_k_ratio=detection_ratio,
            true_poison_indices=true_poison,
            return_metrics=True,
            use_label_signal=args.use_label_signal,
        )

        logger.info(
            f"Detection: precision={det_metrics['precision']:.4f}, "
            f"recall={det_metrics['recall']:.4f}, f1={det_metrics['f1']:.4f}"
        )

        guard.train_defense(
            epochs=args.defense_epochs,
            lr=args.defense_lr,
            weight_decay=args.defense_weight_decay,
            verbose=args.verbose,
            use_clean_graph=args.use_clean_graph,
            use_prune=not args.no_prune,
            min_weight=args.min_weight,
            max_downweight=args.max_downweight,
            # Avoid shape mismatch in some older HeteroGuard versions.
            hard_remove_suspicious=args.hard_remove_suspicious,
            use_trigger_unlearning=args.use_trigger_unlearning,
            attacker=attacker,
            target_class=attacker.target_class,
            unlearn_lambda=args.unlearn_lambda,
            unlearn_samples=args.unlearn_samples,
            target_suppression=args.target_suppression,
            unlearn_exclude_target=not args.include_target_in_unlearning,
        )

        defense_model = guard.get_model()

        defense_asr, defense_asr_clean_acc, _ = measure_asr(
            model=defense_model,
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
        defense_asr = 0.0 if defense_asr is None else float(defense_asr)
        defense_asr_clean_acc = 0.0 if defense_asr_clean_acc is None else float(defense_asr_clean_acc)

        eval_graph = guard.purified_data if guard.purified_data is not None else poison_data
        defense_clean_acc = float(measure_clean_acc(defense_model, eval_graph, target_type, device=device))

        logger.info(f"Defense Clean Acc: {defense_clean_acc:.4f}")
        logger.info(f"Defense ASR: {defense_asr:.4f}")
        logger.info(f"ASR Drop: {poison_asr - defense_asr:+.4f}")

        result["defense"] = {
            "detection_precision": float(det_metrics["precision"]),
            "detection_recall": float(det_metrics["recall"]),
            "detection_f1": float(det_metrics["f1"]),
            "defense_clean_acc": defense_clean_acc,
            "defense_asr": defense_asr,
            "defense_asr_clean_acc": defense_asr_clean_acc,
            "asr_drop": poison_asr - defense_asr,
            "detection_ratio": detection_ratio,
            "use_label_signal": bool(args.use_label_signal),
            "use_clean_graph": bool(args.use_clean_graph),
            "hard_remove_suspicious": bool(args.hard_remove_suspicious),
            "use_trigger_unlearning": bool(args.use_trigger_unlearning),
            "unlearn_lambda": float(args.unlearn_lambda),
            "unlearn_samples": int(args.unlearn_samples),
            "target_suppression": float(args.target_suppression),
        }

    logger.info("=" * 80)
    logger.info("Integrated summary")
    logger.info("=" * 80)
    logger.info(
        f"{args.dataset:8s} {args.model:10s} {args.attack:12s} | "
        f"Clean={clean_acc:.4f}, "
        f"BackdoorClean={backdoored_clean_acc:.4f}, "
        f"CleanTrigASR={clean_trigger_asr:.4f}, "
        f"PoisonASR={poison_asr:.4f}, "
        f"Gap={learned_gap:+.4f}"
    )
    if "defense" in result:
        d = result["defense"]
        logger.info(
            f"Defense | DetF1={d['detection_f1']:.4f}, "
            f"DefClean={d['defense_clean_acc']:.4f}, "
            f"DefASR={d['defense_asr']:.4f}, "
            f"ASRDrop={d['asr_drop']:+.4f}"
        )

    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            args.output_dir,
            f"integrated_{args.dataset}_{args.model}_{args.attack}_r{args.poison_rate:.2f}_seed{args.seed}_{timestamp}.json",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(result), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved result JSON: {out_path}")

    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Integrated in-memory heterogeneous backdoor experiment")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--dataset", type=str, default="ACM", choices=DATASETS)
    parser.add_argument("--model", type=str, default="HAN", choices=MODELS)
    parser.add_argument("--attack", type=str, default="relation", choices=ATTACKS)

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
    parser.add_argument("--poison_save_best", type=str, default="last",
                        choices=["last", "train_loss", "val_acc"])

    parser.add_argument("--poison_rate", type=float, default=0.2)
    parser.add_argument("--trigger_size", type=int, default=10)
    parser.add_argument("--target_class", type=int, default=0)
    parser.add_argument("--trigger_strength", type=float, default=3.0)
    parser.add_argument("--surrogate_epochs", type=int, default=30)
    parser.add_argument("--num_inject", type=int, default=200)

    parser.add_argument("--relation_mode", type=str, default="hybrid",
                        choices=["pure", "hybrid"],
                        help="For relation attack: pure disables target feature boost; hybrid enables it.")
    parser.add_argument("--target_feature_strength", type=float, default=4.0)
    parser.add_argument("--aux_feature_strength", type=float, default=6.0)
    parser.add_argument("--no_aux_clique", action="store_true")

    parser.add_argument("--run_defense", action="store_true")
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--pretrain_lr", type=float, default=0.005)
    parser.add_argument("--defense_epochs", type=int, default=100)
    parser.add_argument("--defense_lr", type=float, default=0.005)
    parser.add_argument("--defense_weight_decay", type=float, default=1e-4)
    parser.add_argument("--detection_ratio", type=float, default=None)
    parser.add_argument("--use_label_signal", action="store_true")
    parser.add_argument("--use_clean_graph", action="store_true")
    parser.add_argument("--no_prune", action="store_true")
    parser.add_argument("--min_weight", type=float, default=0.0)
    parser.add_argument("--max_downweight", type=float, default=1.0)
    parser.add_argument("--hard_remove_suspicious", action="store_true",
                        help="Hard-remove detected suspicious nodes from train_mask. Off by default to avoid older trainer shape issues.")

    parser.add_argument("--use_trigger_unlearning", action="store_true",
                        help="Enable HeteroGuard-Unlearn anti-trigger training.")
    parser.add_argument("--unlearn_lambda", type=float, default=1.0,
                        help="Weight for CE(triggered clean nodes, original labels).")
    parser.add_argument("--unlearn_samples", type=int, default=256,
                        help="Number of remaining clean train nodes used for trigger unlearning.")
    parser.add_argument("--target_suppression", type=float, default=0.1,
                        help="Weight for suppressing target-class probability on triggered clean nodes.")
    parser.add_argument("--include_target_in_unlearning", action="store_true",
                        help="Include target-class nodes in trigger unlearning. Default excludes target-class nodes.")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    logger.info("Configuration:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    run_integrated(args)
