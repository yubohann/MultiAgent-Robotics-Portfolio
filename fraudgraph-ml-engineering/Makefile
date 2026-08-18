.PHONY: validate test compile quality build

PYTHON ?= python

validate:
	$(PYTHON) scripts/validate_repository.py

test:
	$(PYTHON) -m pytest

compile:
	$(PYTHON) -m compileall -q src scripts tests

quality: validate test compile

build:
	$(PYTHON) -m build
