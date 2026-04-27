#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSETS_DIR="${PROJECT_DIR}/assets"
IMG_DIR="${PROJECT_DIR}/img"

source "${SCRIPT_DIR}/latex_env.sh"

# Let LaTeX/BibTeX find .sty/.bst/.bib in assets/ and figures in img/
export TEXINPUTS=".:${ASSETS_DIR}:${IMG_DIR}:${TEXINPUTS:-}"
export BIBINPUTS=".:${ASSETS_DIR}:${BIBINPUTS:-}"
export BSTINPUTS=".:${ASSETS_DIR}:${BSTINPUTS:-}"

cd "${PROJECT_DIR}"

"${LATEXMK_BIN}" \
  -pdf \
  -bibtex \
  -interaction=nonstopmode \
  -halt-on-error \
  -file-line-error \
  -e '$pdflatex = q('"${PDFLATEX_BIN}"' %O %S)' \
  -e '$bibtex = q('"${BIBTEX_BIN}"' %O %B)' \
  -auxdir="${LATEX_BUILD_AUXDIR}" \
  -outdir="${LATEX_BUILD_OUTDIR}" \
  report.tex

cp -f "${LATEX_BUILD_OUTDIR}/report.pdf" "${PROJECT_DIR}/report.pdf"

echo
echo "Build finished."
echo "PDF: ${PROJECT_DIR}/report.pdf"
echo "(also at ${LATEX_BUILD_OUTDIR}/report.pdf)"
