#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_DIR="${1:?run directory required}"
export PYTHONUNBUFFERED=1
echo "Background experiment started: $(date -Is)"
"$ROOT/.venv/bin/python" -u scripts/run_operator_experiment.py --run-dir "$RUN_DIR"
RUN_EXIT=$?
echo "Controller exited: $RUN_EXIT at $(date -Is)"
"$ROOT/.venv/bin/python" scripts/operator_final_report.py "$RUN_DIR" > "$RUN_DIR/final-report.log" 2>&1
REPORT_EXIT=$?
printf 'controller_exit=%s\nreport_exit=%s\nfinished=%s\n' "$RUN_EXIT" "$REPORT_EXIT" "$(date -Is)" > "$RUN_DIR/background-exit.txt"
if [[ "$RUN_EXIT" -ne 0 ]]; then exit "$RUN_EXIT"; fi
exit "$REPORT_EXIT"
