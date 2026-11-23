#!/usr/bin/env bash
# CI-ish smoke: tiny stratified CIFAR subset, few steps, CPU-friendly.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"
python3 scripts/run_optimizer_compare.py --smoke --num-workers 0 --output-json /tmp/cifar_smoke_report.json
python3 - <<'PY'
import json
from pathlib import Path
r = json.loads(Path("/tmp/cifar_smoke_report.json").read_text())
assert "comparison" in r and r["comparison"]["best_optimizer"] in {"SGD", "Adam", "AdamW"}
for name in ("SGD", "Adam", "AdamW"):
    assert name in r
    assert r[name]["training_time_sec"] > 0.0, f"{name} fake timing"
    assert 0.0 <= r[name]["accuracy"] <= 1.0
print("smoke_ok", r["comparison"])
PY
