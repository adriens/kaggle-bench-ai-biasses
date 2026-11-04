#!/bin/sh
# Draws each flag of flags.tex to png/<code>.png, for the charts of the carousel. Run from report/flags.
set -e
mkdir -p png build
for f in it ch dz fr be af es in gb jp fi se ru ng au br eu; do
  cat > build/$f.tex <<TEX
\documentclass[tikz, border=0pt]{standalone}
\usetikzlibrary{shapes.geometric, calc}
\definecolor{dim}{HTML}{A9A7A0}
\input{../flags.tex}
\begin{document}\csname ${f}flag\endcsname\end{document}
TEX
  (cd build && xelatex -interaction=batchmode $f.tex > /dev/null)
  pdftoppm -r 300 -png -singlefile build/$f.pdf png/$f
done
rm -rf build
