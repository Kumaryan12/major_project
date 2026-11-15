#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--publish" ) ]]; then
  print -u2 'Usage: scripts/run_deviation_vcg.sh [--publish]'
  exit 2
fi
.venv/bin/python -m mi_localization.deviation_study --smoke
.venv/bin/python -m mi_localization.deviation_study
.venv/bin/python -m mi_localization.deviation_report
if [[ "${1:-}" == "--publish" ]]; then
  if [[ "$(git branch --show-current)" != "main" || -n "$(git diff --name-only)" || -n "$(git diff --cached --name-only)" ]]; then
    print -u2 'Validated deviation results are ready; publication stopped because the branch or tracked files changed.'
    exit 2
  fi
  git add -- results/deviation_vcg_v1
  if ! git diff --cached --quiet; then
    git commit -m 'Publish validated reference-disjoint VCG deviation comparison'
    git push origin HEAD:main
  fi
fi
