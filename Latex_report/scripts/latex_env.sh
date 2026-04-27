#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TINYTEX_BIN="/scratch/rl182/tex/.TinyTeX/bin/x86_64-linux"
LATEX_ENV_PREFIX="/scratch/rl182/envs/latex"
SCRATCH_TEX_ROOT="/scratch/rl182/tex"
PROJECT_CACHE_ROOT="${SCRATCH_TEX_ROOT}/Project-meme/Latex_report"

if [[ -x "${TINYTEX_BIN}/pdflatex" ]]; then
  PDFLATEX_BIN="${TINYTEX_BIN}/pdflatex"
  BIBTEX_BIN="${TINYTEX_BIN}/bibtex"
  USE_TINYTEX_DEFAULTS=1
elif [[ -x "${LATEX_ENV_PREFIX}/bin/pdflatex" ]]; then
  PDFLATEX_BIN="${LATEX_ENV_PREFIX}/bin/pdflatex"
  BIBTEX_BIN="${LATEX_ENV_PREFIX}/bin/bibtex"
  TEXMFVAR_DIR="${SCRATCH_TEX_ROOT}/texmf-var"
  TEXMFCONFIG_DIR="${SCRATCH_TEX_ROOT}/texmf-config"
  TEXMFHOME_DIR="${SCRATCH_TEX_ROOT}/texmf-home"
  USE_TINYTEX_DEFAULTS=0
else
  echo "LaTeX environment not found." >&2
  echo "Expected pdflatex under ${TINYTEX_BIN} or ${LATEX_ENV_PREFIX}/bin" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ -x "${LATEX_ENV_PREFIX}/bin/latexmk" ]]; then
  LATEXMK_BIN="${LATEX_ENV_PREFIX}/bin/latexmk"
else
  LATEXMK_BIN="${TINYTEX_BIN}/latexmk"
fi

mkdir -p "${PROJECT_CACHE_ROOT}/aux" "${PROJECT_CACHE_ROOT}/out"

if [[ "${USE_TINYTEX_DEFAULTS}" == "0" ]]; then
  mkdir -p "${TEXMFVAR_DIR}" "${TEXMFCONFIG_DIR}" "${TEXMFHOME_DIR}"
fi

export PATH="${TINYTEX_BIN}:$(dirname "${LATEXMK_BIN}"):${PATH}"
if [[ "${USE_TINYTEX_DEFAULTS}" == "0" ]]; then
  export TEXMFVAR="${TEXMFVAR_DIR}"
  export TEXMFCONFIG="${TEXMFCONFIG_DIR}"
  export TEXMFHOME="${TEXMFHOME_DIR}"
else
  unset TEXMFVAR TEXMFCONFIG TEXMFHOME || true
fi
export LATEX_REPORT_DIR="${SCRIPT_DIR}"
export LATEX_BUILD_AUXDIR="${PROJECT_CACHE_ROOT}/aux"
export LATEX_BUILD_OUTDIR="${PROJECT_CACHE_ROOT}/out"
export LATEXMK_BIN
export PDFLATEX_BIN
export BIBTEX_BIN
