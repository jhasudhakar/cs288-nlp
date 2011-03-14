
pdf:	graphs a3.tex a3.bib
	pdflatex a3.tex
	bibtex a3
	pdflatex a3.tex
	pdflatex a3.tex

include graphs/Makefile


clean: cleanGraphs
	rm -f *.aux *.log *.dvi *.blg *.bbl
