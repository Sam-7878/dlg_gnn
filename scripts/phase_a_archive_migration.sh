#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd /mnt/d/_Work/goat_bank/dlg_gnn && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

test -z "${VIRTUAL_ENV:-}" || { echo "Deactivate venv and use a fresh shell"; exit 1; }
test -d ../.venv
test ! -L ../.venv
test -x ../.venv/bin/python

for p in ../.venv_old ../.venv_cuda; do
  test ! -e "$p" && test ! -L "$p" || { echo "Path already exists: $p"; exit 1; }
done

ARCHIVE="$REPO_ROOT/evaluation/benchmark/v2/environment/legacy/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$ARCHIVE"
printf '%s\n' "/mnt/d/_Work/goat_bank/.venv" > "$ARCHIVE/original-venv-path.txt"

echo "[1/6] Capturing environment metadata..."
../.venv/bin/python --version > "$ARCHIVE/python-version.txt" 2>&1
../.venv/bin/python -m pip freeze --all > "$ARCHIVE/requirements-v1-freeze.txt"
../.venv/bin/python -m pip list --format=json > "$ARCHIVE/pip-list-v1.json"
../.venv/bin/python -m pip check > "$ARCHIVE/pip-check.txt" 2>&1 || true

cat << 'EOF' > "$ARCHIVE/dump_sys.py"
import sys, sysconfig, platform, json
data = {
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "version": sys.version,
    "platform": platform.platform(),
    "compiler": platform.python_compiler(),
    "config": sysconfig.get_config_vars()
}
print(json.dumps(data, default=str, indent=2))
EOF
../.venv/bin/python "$ARCHIVE/dump_sys.py" > "$ARCHIVE/python-system.json"
rm "$ARCHIVE/dump_sys.py"

cp ../.venv/pyvenv.cfg "$ARCHIVE/pyvenv.cfg"

echo "[2/6] Capturing git and system state..."
git rev-parse HEAD > "$ARCHIVE/git-commit.txt"
git status --porcelain=v1 > "$ARCHIVE/git-status.txt"
git diff --binary HEAD > "$ARCHIVE/tracked-working-tree.patch"
git ls-files --others --exclude-standard > "$ARCHIVE/untracked-files.txt"
uname -a > "$ARCHIVE/kernel.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "$ARCHIVE/nvidia-smi.txt" 2>&1 || echo "PROBE_FAILED" >> "$ARCHIVE/nvidia-smi.txt"
fi

echo "[3/6] Archiving ../.venv to $ARCHIVE/venv-before-rename.tar.gz (this may take a minute)..."
tar -czf "$ARCHIVE/venv-before-rename.tar.gz" -C /mnt/d/_Work/goat_bank .venv

echo "[4/6] Verifying archive integrity and checksum..."
tar -tzf "$ARCHIVE/venv-before-rename.tar.gz" | head -n 50 > "$ARCHIVE/archive-members.txt"
sha256sum "$ARCHIVE/venv-before-rename.tar.gz" > "$ARCHIVE/venv-before-rename.tar.gz.sha256"
sha256sum -c "$ARCHIVE/venv-before-rename.tar.gz.sha256"

echo "[5/6] Atomically renaming ../.venv to ../.venv_old..."
test ! -e ../.venv_old
test ! -L ../.venv_old
mv -nT -- ../.venv ../.venv_old
test ! -e ../.venv
test ! -L ../.venv
test -d ../.venv_old

echo "[6/6] Writing migration receipt..."
printf '%s\n' 'ARCHIVED_AND_RENAMED; ../.venv_old is not an activated runtime' > "$ARCHIVE/migration-receipt.txt"

echo "PHASE_A_SUCCESS"
printf "Archive location: %s\n" "$ARCHIVE"
