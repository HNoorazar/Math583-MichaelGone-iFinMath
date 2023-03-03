# Simple makefile with common rules.

makefiles_dir = .makefiles
include $(makefiles_dir)/rst2html.mk

# This gets the version of python that mercurial uses
PYTHON="$$(shebang=$$(head -n1 $$(type -p hg));echo $${shebang:2})"
PYTHON=python
HG=`which hg`
MMF_SETUP=$(shell pwd)/src/mmf_setup
MMF_SETUP=$(shell python -c "import os.path, mmf_setup;print(os.path.dirname(mmf_setup.__file__))")

nbinit.py: make_nbinit.py src/mmf_setup/_data/nbthemes/mmf.*
	python make_nbinit.py

help:
	@echo 'Commonly used make targets:'
	@echo '  test - run all tests in the automatic test suite'

test-cocalc:
	cd tests && MMF_SETUP=$(MMF_SETUP) $(PYTHON) run-tests.py --with-hg=$(HG) test-cocalc*.t $(TESTFLAGS)

test-hg:
	cd tests && MMF_SETUP=$(MMF_SETUP) $(PYTHON) run-tests.py --with-hg=$(HG) $(TESTFLAGS)

test-py:
	PY_IGNORE_IMPORTMISMATCH=1 pytest

test: test-hg test-py

README_CHANGES.html: README.rst CHANGES.txt
	cat $^ | rst2html.py > $@

clean:
	rm -rf .nox
	rm -rf .pytest_cache
	rm -rf mmf_setup.egg-info
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "__pycache__" -type d -delete
	rm README_CHANGES.html

.PHONY: help test-cocalc test-hg test-py test clean
