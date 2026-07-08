# Build paper.pdf from docs/paper.tex
.PHONY: pdf figures clean

figures:
	python generate_paper_figures.py

pdf:
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex

pdf-full: figures pdf

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.toc paper.pdf
