# Variables
PYTHON_INTERPRETER = /Users/thibaultgillard/miniconda3/envs/poc_to_prod_capstone_m2/bin/python
# app path predict/predict/app.py
FLASK_APP = predict.predict.app

# Create Conda environment environment.yml
install:
	conda env create -f environment.yml

# Activate Conda environment
activate:
	conda activate poc_to_prod_capstone_m2

# Deactivate Conda environment
deactivate:
	conda deactivate

# Run the Flask app
run:
	FLASK_APP=$(FLASK_APP) $(PYTHON_INTERPRETER) -m flask run

.PHONY: predict-test-unitest preprocessing-test-unitest train-test-unitest

predict-test-unitest:
	$(PYTHON_INTERPRETER) -m unittest discover -s predict/tests -p 'test_*.py'

preprocessing-test-unitest:
	$(PYTHON_INTERPRETER) -m unittest discover -s preprocessing/tests -p 'test_*.py'

train-test-unitest:
	$(PYTHON_INTERPRETER) -m unittest discover -s train/tests -p 'test_*.py'

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  install      to create Conda environment"
	@echo "  activate     to activate Conda environment"
	@echo "  deactivate   to deactivate Conda environment"
	@echo "  run          to run the Flask app"
	@echo "  test         to run tests"
	@echo "  help         to display this help message"
