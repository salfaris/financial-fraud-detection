report:
	quarto render docs/report/report.qmd --to html
	open docs/report/report.html

# .PHONY is used to ensure that the target is not associated with a file
.PHONY: report
