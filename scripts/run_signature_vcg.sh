#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--publish" ) ]]; then
  print -u2 'Usage: scripts/run_signature_vcg.sh [--publish]'
  exit 2
fi
.venv/bin/python -m mi_localization.signature_study --smoke
.venv/bin/python -m mi_localization.signature_study
.venv/bin/python -m mi_localization.signature_report
if [[ "${1:-}" == "--publish" ]]; then
  if [[ "$(git branch --show-current)" != "main" || -n "$(git diff --name-only)" || -n "$(git diff --cached --name-only)" ]]; then
    print -u2 'Validated signature results are ready; publication stopped because branch/tracked files changed.'
    exit 2
  fi
  git add -- results/signature_vcg_v1
  if ! git diff --cached --quiet; then
    git commit -m 'Publish validated ordered VCG signature comparison'
    git push origin HEAD:main
  fi
fi
