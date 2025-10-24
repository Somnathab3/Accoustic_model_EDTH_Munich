PY?=python
PIP?=pip

env:
	$(PY) -m venv .venv && . .venv/bin/activate && $(PIP) install -U pip && $(PIP) install -r requirements.txt

hf-login:
	huggingface-cli login

data:
	. .venv/bin/activate && $(PY) scripts/download_data.py --out-dir data

prep:
	. .venv/bin/activate && $(PY) scripts/prepare_dataset.py --in-csv data/raw/metadata_train.csv --out-dir data/processed

train:
	. .venv/bin/activate && $(PY) -m src.adrone.train --config configs/train.yaml

serve:
	. .venv/bin/activate && uvicorn src.adrone.serve.app:app --host 0.0.0.0 --port 8000 --reload
