#!/usr/bin/env bash
# Resume the bounded A05 DGraphFin exact AnomalyDAE campaign.
set -euo pipefail
cd /mnt/d/_work/goat_bank/dlg_gnn
PY=/mnt/d/_work/goat_bank/.venv_cuda/bin/python
RAW=outputs/benchmark/a05_anomalydae_large/raw
LOG=outputs/benchmark/a05_anomalydae_large/continuation.log
mkdir -p "$RAW"
export CUDA_VISIBLE_DEVICES=0

exec 9>outputs/benchmark/a05_anomalydae_large/continuation.lock
if ! flock -n 9; then
    printf 'A05 DGraphFin campaign is already running.\n'
    exit 0
fi
trap 'code=$?; printf "A05 DGraphFin campaign exit code: %s\\n" "$code" >> "$LOG"' EXIT

finish_release() {
    "$PY" evaluation/benchmark/v2/scripts/a05_finalize_anomalydae.py | tee -a "$LOG"
    "$PY" evaluation/benchmark/v2/scripts/a05_build_release.py | tee -a "$LOG"
    "$PY" evaluation/benchmark/v2/scripts/a05_write_manuscript.py | tee -a "$LOG"
    (cd docs/papers/_42_01_Benchmark_PrePrints && latexmk -pdf -interaction=nonstopmode -halt-on-error -quiet DLG-Benchmark_Preprint_A05.tex) >> "$LOG" 2>&1
    "$PY" evaluation/benchmark/v2/scripts/90_validate_results_a05_journal.py | tee -a "$LOG"
    touch outputs/benchmark/a05_anomalydae_large/campaign.done
}

for seed in 42 43 44 45 46; do
    printf 'Starting DGraphFin AnomalyDAE seed %s\n' "$seed" | tee -a "$LOG"
    set +e
    timeout 86400 "$PY" evaluation/benchmark/v2/scripts/a05_run_anomalydae_large.py DGraphFin --seed "$seed" 2>&1 | tee -a "$LOG"
    run_code=${PIPESTATUS[0]}
    set -e
    if (( run_code == 124 )); then
        "$PY" - "$seed" <<'PY'
import json
import sys
from pathlib import Path

seed = int(sys.argv[1])
base = Path('outputs/benchmark/a05_anomalydae_large')
path = base / 'raw' / f'DGraphFin__AnomalyDAE__seed{seed}.json'
record = json.loads(path.read_text())
progress = base / 'checkpoints' / f'DGraphFin__seed{seed}.pt.progress.json'
record.update(status='UNSUPPORTED_OPERATIONAL_TIMEOUT', timeout_budget_sec=86400,
              completed_epochs_at_timeout=(json.loads(progress.read_text())['completed_epochs']
                                           if progress.exists() else 0))
path.write_text(json.dumps(record, indent=2, default=str) + '\n')
PY
        printf 'DGraphFin seed %s reached the 24-hour execution guard.\n' "$seed" | tee -a "$LOG"
        finish_release
        exit 0
    fi
    if (( run_code != 0 )); then
        printf 'DGraphFin seed %s failed with exit code %s.\n' "$seed" "$run_code" | tee -a "$LOG"
        exit "$run_code"
    fi
done

finish_release
printf 'A05 DGraphFin continuation finished with journal gate PASS.\n' | tee -a "$LOG"
