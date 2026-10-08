#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON:-python3}" "$repo_root/scripts/reproduce_project.py" --project dlg_gnn --mode "${1:-verify}" "${@:2}"
