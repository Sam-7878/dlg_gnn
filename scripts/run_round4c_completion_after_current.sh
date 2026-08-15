#!/usr/bin/env bash
set -euo pipefail

wait_pid="${1:?current Round 4C matrix PID is required}"
while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 60
done

cd /mnt/d/_Work/goat_bank/dlg_gnn
export PYTHONPATH=src
exec ../.venv/bin/python -m gog_fraud.pipelines.run_sci_round4c_completion \
  --config configs/benchmark/sci_round4c_production.yaml \
  --prior-dgraph-active-sec 63064
