#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--publish" ) ]]; then
  print -u2 'Usage: scripts/run_tabpfn_v2.sh [--publish]'
  exit 2
fi
.venv/bin/python -m mi_localization.tabpfn_backend
.venv/bin/python -m mi_localization.tabpfn_study
.venv/bin/python -m mi_localization.tabpfn_report
if [[ "${1:-}" == "--publish" ]]; then
  if [[ "$(git branch --show-current)" != "main" || -n "$(git diff --name-only)" || -n "$(git diff --cached --name-only)" ]]; then
    print -u2 'Validated TabPFN results are ready. Automatic publication stopped because the branch or tracked working tree changed.'
    exit 2
  fi
  git add -- results/tabpfn_v2_v1
  if ! git diff --cached --quiet; then
    git commit -m 'Publish validated pinned TabPFN-v2 patient benchmark'
    git push origin HEAD:main
  fi
fi
