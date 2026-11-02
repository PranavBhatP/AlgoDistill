#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "=== Building Cover Page ==="
pdflatex -interaction=nonstopmode cover_page.tex

echo "=== Building IEEE Access Paper (Pass 1) ==="
pdflatex -interaction=nonstopmode main_paper.tex

echo "=== Building IEEE Access Paper (Pass 2 for Cross-References) ==="
pdflatex -interaction=nonstopmode main_paper.tex

echo "=== Merging Cover Page and Research Paper ==="
pdfunite cover_page.pdf main_paper.pdf Final_Research_Report_IEEE_Access.pdf

echo "=== Copying Final Artifacts ==="
cp Final_Research_Report_IEEE_Access.pdf ../Final_Research_Report_IEEE_Access.pdf
cp Final_Research_Report_IEEE_Access.pdf ../231IT049_IT488_ProjectReport.pdf
mkdir -p ../docs
cp Final_Research_Report_IEEE_Access.pdf ../docs/IT488_Team9_Final_Research_Report.pdf

echo "=== Build Complete! ==="
pdfinfo Final_Research_Report_IEEE_Access.pdf
