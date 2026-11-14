#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--publish" ) ]]; then
  print -u2 'Usage: scripts/run_paired_ecg_vcg.sh [--publish]'
  exit 2
fi
.venv/bin/python -m mi_localization.paired_bank
.venv/bin/python -m mi_localization.paired_study
.venv/bin/python -m mi_localization.paired_report
if [[ "${1:-}" == "--publish" ]]; then
  if [[ "$(git branch --show-current)" != "main" || -n "$(git diff --name-only)" || -n "$(git diff --cached --name-only)" ]]; then
    print -u2 'Validated results are ready. Automatic push stopped because the branch or tracked working tree changed.'
    exit 2
  fi
  git add -- results/neural_paired_ecg_v1/complete
  if ! git diff --cached --quiet; then
    git commit -m 'Publish validated measured ECG plus VCG fusion study'
    git push origin HEAD:main
  fi
fi
