# NeuScale CIFAR-10 — honest eval harness

Distributed image-classification demo (DeiT-tiny + Hugging Face `Accelerator`) with a **production-truthful** CIFAR-10 evaluation contract. Replaces the previous 50-image toy loop that evaluated on the shuffled train loader and hardcoded `training_time = 1.0`.

## Eval contract

| Rule | Detail |
|------|--------|
| Data | Official CIFAR-10 (Hugging Face `uoft-cs/cifar10`, or local `torchvision` extract under `data/cifar10/`); optional **stratified** subset via `--train-per-class` / `--test-per-class` |
| Splits | Fixed **train / val / test**. Val is carved from the official train set (stratified). The official test set is never used for training or model selection. |
| Seeds | `--seed` controls Python / NumPy / Torch RNG and stratified draws |
| Metrics | Top-1 accuracy reported **only on held-out test** |
| Timing | Wall-clock seconds and samples/sec measured around the training loop under `Accelerator` (no hardcoded timing) |
| Optimizer ranking | Best optimizer is chosen **inside** the comparison loop by held-out test accuracy |

Val exists for future early-stopping / hyperparameter work; this epic’s reported ranking uses **test** accuracy only (same split every seed).

## Quick start

```bash
pip install -r requirements.txt

# Full CIFAR-10 (downloads via Hugging Face Hub; local torchvision extract also works)
python scripts/run_optimizer_compare.py --epochs 2

# Large stratified subset example
python scripts/run_optimizer_compare.py --train-per-class 500 --test-per-class 100

# CI-ish smoke (tiny subset, few steps)
bash scripts/smoke_test.sh
# or:
python scripts/run_optimizer_compare.py --smoke --num-workers 0
```

## Layout

- `cifar_eval/data.py` — loaders, stratified subsets, seed control
- `cifar_eval/train_eval.py` — train, held-out eval, real timing, correct best-optimizer selection
- `scripts/run_optimizer_compare.py` — CLI entrypoint
- `notebook.ipynb` — same contract, calls into `cifar_eval`

## Out of scope (later epics)

Multi-GPU launch matrix, SSL/RBM ablations, CIFAR-100, full MLOps packaging, LangGraph/E2B wiring.
