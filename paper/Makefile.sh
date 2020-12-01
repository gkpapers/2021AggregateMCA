TEX="pdflatex"
BIBTEX="bibtex"
FILE="aggregating-unstable-derivatives"

default:
	${TEX} ${FILE} &&\
	${TEX} ${FILE} &&\
	${BIBTEX} ${FILE} &&\
	${TEX} ${FILE} &&\
	${TEX} ${FILE}

clean:
	rm -f ${FILE}{.bcf,.blg,.bbl,.log,.aux,.out,.fdb_latexmk,.fls,.run.xml,.synctex.gz} &&\
	rm Makefile

cleanall:
	make clean &&\
	rm -f ${FILE}.pdf
