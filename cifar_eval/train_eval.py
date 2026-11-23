"""Train / evaluate with real wall-clock timing and correct optimizer selection."""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, Optional, Tuple

import evaluate
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification

from .data import set_seed

OptimizerFactory = Callable[[Any], optim.Optimizer]

DEFAULT_OPTIMIZER_CONFIGS: Dict[str, OptimizerFactory] = {
    "SGD": lambda p: optim.SGD(p, lr=0.01, momentum=0.9, weight_decay=0.0),
    "Adam": lambda p: optim.Adam(p, lr=0.001, weight_decay=1e-4),
    "AdamW": lambda p: optim.AdamW(p, lr=0.001, weight_decay=1e-4),
}

OPTIMIZER_HYPERPARAMS = {
    "SGD": {"learning_rate": 0.01, "weight_decay": 0.0},
    "Adam": {"learning_rate": 0.001, "weight_decay": 1e-4},
    "AdamW": {"learning_rate": 0.001, "weight_decay": 1e-4},
}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    accelerator: Accelerator,
) -> float:
    """Top-1 accuracy on ``dataloader`` (intended: held-out test only)."""
    metric = evaluate.load("accuracy")
    model.eval()
    for inputs, labels in dataloader:
        outputs = model(pixel_values=inputs).logits
        preds = outputs.argmax(dim=1)
        preds, labels = accelerator.gather_for_metrics((preds, labels))
        metric.add_batch(
            predictions=preds.detach().cpu().numpy(),
            references=labels.detach().cpu().numpy(),
        )
    return float(metric.compute()["accuracy"])


def train_one_optimizer(
    model_name: str,
    initial_state_dict: Dict[str, torch.Tensor],
    opt_fn: OptimizerFactory,
    train_loader: DataLoader,
    test_loader: DataLoader,
    accelerator: Accelerator,
    num_epochs: int = 2,
    max_steps: Optional[int] = None,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Fine-tune from ``initial_state_dict``, time wall-clock under Accelerator,
    and report accuracy **only** on ``test_loader``.
    """
    criterion = criterion or nn.CrossEntropyLoss()
    model = AutoModelForImageClassification.from_pretrained(
        model_name, num_labels=10, ignore_mismatched_sizes=True
    )
    model.load_state_dict(initial_state_dict)

    optimizer = opt_fn(model.parameters())
    model, optimizer, prepared_train, prepared_test = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    model.train()
    samples_seen = 0
    steps = 0
    t0 = time.perf_counter()

    for _ in range(num_epochs):
        for inputs, labels in prepared_train:
            outputs = model(pixel_values=inputs).logits
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            samples_seen += int(labels.shape[0])
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        if max_steps is not None and steps >= max_steps:
            break

    # Sync before stopping the clock so multi-process runs measure real work.
    accelerator.wait_for_everyone()
    elapsed = time.perf_counter() - t0

    accuracy = evaluate_model(model, prepared_test, accelerator)
    throughput = samples_seen / elapsed if elapsed > 0 else 0.0

    return {
        "accuracy": accuracy,
        "training_time_sec": elapsed,
        "samples_per_sec": throughput,
        "samples_seen": float(samples_seen),
        "steps": float(steps),
    }


def compare_optimizers(
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    model_name: str = "facebook/deit-tiny-patch16-224",
    num_epochs: int = 2,
    max_steps: Optional[int] = None,
    seed: int = 42,
    optimizer_configs: Optional[Dict[str, OptimizerFactory]] = None,
    accelerator: Optional[Accelerator] = None,
) -> Dict[str, Any]:
    """
    Run each optimizer from the same init; pick the best by held-out test accuracy
    **inside** the comparison loop (fixes the notebook bug that only checked the last run).
    """
    set_seed(seed)
    accelerator = accelerator or Accelerator()
    configs = optimizer_configs or DEFAULT_OPTIMIZER_CONFIGS

    base = AutoModelForImageClassification.from_pretrained(
        model_name, num_labels=10, ignore_mismatched_sizes=True
    )
    # Deep-copy so later mutates (prepare / train) cannot corrupt the shared init.
    initial_state_dict = copy.deepcopy(base.state_dict())
    del base

    optimizer_summary: Dict[str, Dict[str, float]] = {}
    training_report: Dict[str, Any] = {}
    best_optimizer: Optional[str] = None
    best_accuracy = -1.0

    for name, opt_fn in configs.items():
        set_seed(seed)  # identical start per optimizer
        result = train_one_optimizer(
            model_name=model_name,
            initial_state_dict=initial_state_dict,
            opt_fn=opt_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            accelerator=accelerator,
            num_epochs=num_epochs,
            max_steps=max_steps,
        )
        optimizer_summary[name] = OPTIMIZER_HYPERPARAMS.get(
            name, {"learning_rate": None, "weight_decay": None}
        )
        training_report[name] = result

        # Selection must live inside the loop — previous notebook bug left this outside.
        if result["accuracy"] > best_accuracy:
            best_accuracy = result["accuracy"]
            best_optimizer = name

    training_report["comparison"] = {
        "best_optimizer": best_optimizer,
        "best_test_accuracy": best_accuracy,
        "metric": "held_out_test_top1",
    }
    training_report["optimizer_summary"] = optimizer_summary
    return training_report


def format_report(report: Dict[str, Any], num_epochs: int) -> str:
    lines = ["", "Training Report Summary (held-out test)"]
    for name, r in report.items():
        if name in ("comparison", "optimizer_summary"):
            continue
        lines.append(
            f"\n{name} Optimizer:\n"
            f"- Test accuracy: {r['accuracy'] * 100:.2f}%\n"
            f"- Wall-clock: {r['training_time_sec']:.2f}s\n"
            f"- Throughput: {r['samples_per_sec']:.1f} samples/s"
        )
    best = report["comparison"]["best_optimizer"]
    lines.append(
        f"\nBest Optimizer after {num_epochs} epoch(s) "
        f"(by held-out test accuracy): {best}"
    )
    return "\n".join(lines)
