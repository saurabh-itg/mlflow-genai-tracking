.PHONY: install prompts experiments register regression ui demo clean

PY ?= python3

install:
	$(PY) -m pip install -r requirements.txt

prompts:
	cd scripts && $(PY) 01_log_prompts.py

experiments:
	cd scripts && $(PY) 02_run_experiments.py

register:
	cd scripts && $(PY) 03_register_model.py

regression:
	cd scripts && $(PY) 04_regression_check.py

demo: prompts experiments register regression

ui:
	mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

clean:
	rm -rf mlruns.db mlartifacts best_run_id.txt champion_config.json
