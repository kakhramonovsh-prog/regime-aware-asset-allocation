# Reproduction pipeline. On Windows without make, run the same commands
# directly (see README "Reproduction").

PYTHON ?= .venv/Scripts/python
ifeq ($(OS),)
PYTHON := .venv/bin/python
endif

.PHONY: all data analysis backtest paper test

all: data analysis backtest paper

data:
	$(PYTHON) scripts/download_data.py

analysis:
	$(PYTHON) scripts/run_analysis.py

backtest:
	$(PYTHON) scripts/run_backtest.py

test:
	$(PYTHON) -m pytest tests/

# Paper build gate: regenerate every number from stored results, compile
# under halt-on-error, fail on undefined citations/references/commands or
# significant overfull boxes, save the log, produce the PDF.
.PHONY: paper-build paper-pages
paper-build:
	$(PYTHON) scripts/build_paper.py

paper-pages: paper-build
	$(PYTHON) scripts/render_paper_pages.py

# `make paper` builds and renders for inspection.
paper: paper-pages
