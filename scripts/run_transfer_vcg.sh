#!/bin/zsh
# Run under caffeinate -i; all completed fits resume from their existing scope.
set -euo pipefail
cd "${0:A:h:h}"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--publish" ) ]]; then
  print -u2 'Usage: scripts/run_transfer_vcg.sh [--publish]'
  exit 2
fi
.venv/bin/python -m mi_localization.transfer
.venv/bin/python -m mi_localization.transfer_report
if [[ "${1:-}" == "--publish" ]]; then
  # Never include another task's edits or publish from an unexpected branch.
  if [[ "$(git branch --show-current)" != "main" || -n "$(git diff --name-only)" || -n "$(git diff --cached --name-only)" ]]; then
    print -u2 'Validated report is ready. Automatic result push stopped because the branch or tracked working tree changed.'
    exit 2
  fi
  git add -- results/neural_transfer_v1/complete
  if ! git diff --cached --quiet; then
    git commit -m 'Publish validated exploratory masked-encoder transfer comparison'
    git push origin HEAD:main
  fi
fi
