#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = real-time-monitoring-system
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 DSML

all:
	requirements clean lint

.PHONY: preprocess
preprocess:
	python -m DSML.preproc

.PHONY: train
train:
	python -m DSML.train
