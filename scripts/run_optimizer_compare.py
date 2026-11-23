#!/usr/bin/env python3
"""Compare SGD / Adam / AdamW on CIFAR-10 with held-out test metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without install.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import logging as hf_logging

from cifar_eval.data import build_cifar10_loaders, set_seed
from cifar_eval.train_eval import compare_optimizers, format_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/cifar10")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument(
        "--train-per-class",
        type=int,
        default=None,
        help="Stratified subset size per class from official train (None = full).",
    )
    p.add_argument(
        "--test-per-class",
        type=int,
        default=None,
        help="Stratified subset size per class from official test (None = full).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional per-optimizer step cap (smoke / debug).",
    )
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--model-name", default="facebook/deit-tiny-patch16-224")
    p.add_argument("--output-json", default=None, help="Optional path to write report JSON.")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny stratified subset + 1 epoch + few steps for CI-ish checks.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    hf_logging.set_verbosity_error()
    set_seed(args.seed)

    train_per_class = args.train_per_class
    test_per_class = args.test_per_class
    epochs = args.epochs
    max_steps = args.max_steps
    image_size = args.image_size
    batch_size = args.batch_size

    if args.smoke:
        train_per_class = train_per_class or 8
        test_per_class = test_per_class or 4
        epochs = min(epochs, 1)
        max_steps = max_steps or 2
        image_size = min(image_size, 224)
        batch_size = min(batch_size, 16)

    train_loader, val_loader, test_loader, sizes = build_cifar10_loaders(
        data_root=args.data_root,
        batch_size=batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=image_size,
        val_fraction=args.val_fraction,
        train_per_class=train_per_class,
        test_per_class=test_per_class,
    )
    # Val is available for future early-stopping; selection metric is held-out test only.
    _ = val_loader

    print(
        f"Splits — train: {sizes.train}, val: {sizes.val}, test: {sizes.test} "
        f"(seed={args.seed})"
    )
    print("Eval contract: metrics reported on held-out test only.")

    report = compare_optimizers(
        train_loader,
        test_loader,
        model_name=args.model_name,
        num_epochs=epochs,
        max_steps=max_steps,
        seed=args.seed,
    )
    print(format_report(report, epochs))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            k: (v if not isinstance(v, dict) else dict(v)) for k, v in report.items()
        }
        out.write_text(json.dumps(serializable, indent=2) + "\n")
        print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
