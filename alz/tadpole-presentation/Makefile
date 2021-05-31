project = presentation

project : $(project).tex 
	xelatex -shell-escape $(project)
	bibtex $(project)
	xelatex -shell-escape $(project)
	xelatex -shell-escape $(project)


clean:
	rm -f $(project).aux $(project).bbl $(project).blg $(project).dvi $(project).fdb_latexmk $(project).log $(project).nav $(project).out $(project).pdf $(project).snm $(project).toc missfont.log

fresh:
	make clean && make
