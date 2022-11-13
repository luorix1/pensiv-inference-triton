PYTHON=3.9
BASENAME=$(shell basename $(CURDIR))

env:
	conda create -n $(BASENAME)  python=$(PYTHON)

setup:
	pip install -r requirements.txt
	pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
	pre-commit install

format:
	black .
	isort .

lint:
	pytest src --flake8 --pylint --mypy

utest:
	PYTHONPATH=src pytest test/utest --cov=src --cov-report=html --cov-report=term --cov-config=setup.cfg

triton:
	docker run --gpus 0,1,2,3 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
		-v $(PWD)/model_repository:/models nvcr.io/nvidia/tritonserver:21.07-py3 \
		tritonserver --model-repository=/models

api:
	PYTHONPATH=src uvicorn api.main:app --reload --host 0.0.0.0 --port 8888 --workers 5 --use-colors --log-level debug

locust:
	locust -f src/locust/locustfile.py APIUser