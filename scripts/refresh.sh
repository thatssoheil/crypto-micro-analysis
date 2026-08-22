#!/usr/bin/env bash
# ============================================================
# crypto-micro-analysis refresh runner (on-demand, stateless)
#
# Fetches the latest data and re-runs engine + audit.
# Sources append daily: Bitstamp, Yahoo, alternative.me.
#
# STATELESS: results print to stdout only, never saved. The only
# files written are the dataset CSVs (the input). Does NOT
# git-pull, push, or schedule anything - run it when asked.
#
# Usage:  scripts/refresh.sh            # fetch + engine + audit
#         scripts/refresh.sh --check    # status only
#
# Exit codes: 0 = ok, 1 = failed (check stderr)
# ============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PYTHON:-$REPO_DIR/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "No venv at $REPO_DIR/.venv - run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
    echo "=== crypto-micro-analysis status (stateless) ==="
    git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo "not a git repo"
    echo "last data refresh: $(stat -c %y "$REPO_DIR/data/micro_dataset/manifest.json" 2>/dev/null | cut -d. -f1)"
    LAST=$("$PY" -c "
import pandas as pd
df = pd.read_csv('$REPO_DIR/data/micro_dataset/btcusd_daily.csv')
print(df['ts'].iloc[-1][:10], round(float(df['close'].iloc[-1]), 2))
" 2>/dev/null || echo n/a)
    echo "BTC data through: $LAST"
    echo "(no saved results - run scripts/refresh.sh to generate the current verdict)"
    exit 0
fi

echo "=== [1/3] fetch latest data ==="
"$PY" "$REPO_DIR/strategies/build_micro_dataset.py" 2>&1 | tail -7

echo "=== [2/3] micro regime engine ==="
"$PY" "$REPO_DIR/strategies/micro_regime.py" 2>&1 | tail -12

echo "=== [3/3] audit ==="
"$PY" "$REPO_DIR/strategies/audit_micro.py" 2>&1 | tail -4

echo
echo "=== DONE. Verdict above (printed to stdout, not saved). ==="
