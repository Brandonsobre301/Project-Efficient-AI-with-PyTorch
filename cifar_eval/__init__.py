"""Honest CIFAR-10 eval harness for NeuScale optimizer comparison."""

from .data import CIFAR_CLASSES, build_cifar10_loaders, set_seed
from .train_eval import compare_optimizers, evaluate_model, train_one_optimizer

__all__ = [
    "CIFAR_CLASSES",
    "build_cifar10_loaders",
    "set_seed",
    "compare_optimizers",
    "evaluate_model",
    "train_one_optimizer",
]
