#!/usr/bin/env bash
xelatex paper_of_the_project.tex
bibtex paper_of_the_project
xelatex paper_of_the_project.tex
xelatex paper_of_the_project.tex
rm -f *.aux *.bbl *.blg *.log *.out *.toc