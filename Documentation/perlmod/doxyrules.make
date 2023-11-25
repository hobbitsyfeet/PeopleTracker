DOXY_EXEC_PATH = K:/Github/PeopleTracker/Documentation
DOXYFILE = K:/Github/PeopleTracker/Documentation/-
DOXYDOCS_PM = K:/Github/PeopleTracker/Documentation/perlmod/DoxyDocs.pm
DOXYSTRUCTURE_PM = K:/Github/PeopleTracker/Documentation/perlmod/DoxyStructure.pm
DOXYRULES = K:/Github/PeopleTracker/Documentation/perlmod/doxyrules.make
DOXYLATEX_PL = K:/Github/PeopleTracker/Documentation/perlmod/doxylatex.pl
DOXYLATEXSTRUCTURE_PL = K:/Github/PeopleTracker/Documentation/perlmod/doxylatex-structure.pl
DOXYSTRUCTURE_TEX = K:/Github/PeopleTracker/Documentation/perlmod/doxystructure.tex
DOXYDOCS_TEX = K:/Github/PeopleTracker/Documentation/perlmod/doxydocs.tex
DOXYFORMAT_TEX = K:/Github/PeopleTracker/Documentation/perlmod/doxyformat.tex
DOXYLATEX_TEX = K:/Github/PeopleTracker/Documentation/perlmod/doxylatex.tex
DOXYLATEX_DVI = K:/Github/PeopleTracker/Documentation/perlmod/doxylatex.dvi
DOXYLATEX_PDF = K:/Github/PeopleTracker/Documentation/perlmod/doxylatex.pdf

.PHONY: clean-perlmod
clean-perlmod::
	rm -f $(DOXYSTRUCTURE_PM) \
	$(DOXYDOCS_PM) \
	$(DOXYLATEX_PL) \
	$(DOXYLATEXSTRUCTURE_PL) \
	$(DOXYDOCS_TEX) \
	$(DOXYSTRUCTURE_TEX) \
	$(DOXYFORMAT_TEX) \
	$(DOXYLATEX_TEX) \
	$(DOXYLATEX_PDF) \
	$(DOXYLATEX_DVI) \
	$(addprefix $(DOXYLATEX_TEX:tex=),out aux log)

$(DOXYRULES) \
$(DOXYMAKEFILE) \
$(DOXYSTRUCTURE_PM) \
$(DOXYDOCS_PM) \
$(DOXYLATEX_PL) \
$(DOXYLATEXSTRUCTURE_PL) \
$(DOXYFORMAT_TEX) \
$(DOXYLATEX_TEX): \
	$(DOXYFILE)
	cd $(DOXY_EXEC_PATH) ; doxygen "$<"

$(DOXYDOCS_TEX): \
$(DOXYLATEX_PL) \
$(DOXYDOCS_PM)
	perl -I"$(<D)" "$<" >"$@"

$(DOXYSTRUCTURE_TEX): \
$(DOXYLATEXSTRUCTURE_PL) \
$(DOXYSTRUCTURE_PM)
	perl -I"$(<D)" "$<" >"$@"

$(DOXYLATEX_PDF) \
$(DOXYLATEX_DVI): \
$(DOXYLATEX_TEX) \
$(DOXYFORMAT_TEX) \
$(DOXYSTRUCTURE_TEX) \
$(DOXYDOCS_TEX)

$(DOXYLATEX_PDF): \
$(DOXYLATEX_TEX)
	pdflatex -interaction=nonstopmode "$<"

$(DOXYLATEX_DVI): \
$(DOXYLATEX_TEX)
	latex -interaction=nonstopmode "$<"
