#!/usr/bin/env bash
pdflatex paper_of_the_project.tex
bibtex paper_of_the_project
pdflatex paper_of_the_project.tex
pdflatex paper_of_the_project.tex
rm -f *.aux *.bbl *.blg *.log *.out *.toc

