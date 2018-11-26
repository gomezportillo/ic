# Basado en la documentación LaTeX de github.com/erseco

all: doc clean

doc:
	cd text; \
	pdflatex -shell-escape -synctex=1 -interaction=nonstopmode -output-directory=../ project.tex; \
	pdflatex -shell-escape -synctex=1 -interaction=nonstopmode -output-directory=../ project.tex; \

.PHONY: clean

clean:
	@echo "Limpiando..."
	@rm -f *.aux *.lof *.log *.lol *.lot *.out *.synctex.gz *.toc

install:
	sudo apt-get install texlive texlive-latex-extra texlive-lang-spanish xzdec
	tlmgr init-usertree
	tlmgr install xcolor
