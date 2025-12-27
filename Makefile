.PHONY: help install test build deploy clean build-inference deploy-inference test-inference run-api generate-data explore-data test-dataset train-model evaluate-model optimize-model benchmark-latency

help:
	@echo "AutoShield-K8s Commands:"
	@echo "  install     Install dependencies"
	@echo "  test        Run all tests"
	@echo "  build       Build Docker images"
	@echo "  build-inference  Build inference service image"
	@echo "  deploy      Deploy to Kubernetes"
	@echo "  deploy-inference Deploy inference service to Kubernetes"
	@echo "  test-inference   Run inference integration tests"
	@echo "  run-api     Start FastAPI inference API server"
	@echo "  monitor     Start monitoring stack"
	@echo "  generate-data Generate training dataset"
	@echo "  explore-data  Start data exploration notebook"
	@echo "  test-dataset  Test dataset quality"
	@echo "  train-model   Train CNN-LSTM model"
	@echo "  evaluate-model Evaluate model"
	@echo "  optimize-model Optimize model for inference"
	@echo "  benchmark-latency Benchmark inference latency"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ --cov=autoshield --cov-report=html

build:
	docker build -f docker/feature-extractor.Dockerfile -t autoshield/feature-extractor:latest .
	docker build -f docker/inference-service.Dockerfile -t autoshield/inference:latest .

build-inference:
	@echo "Building inference service..."
	docker build -f docker/inference-service.Dockerfile -t autoshield/inference:latest .

deploy:
	kubectl apply -f deployment/

deploy-inference:
	@echo "Deploying inference service..."
	kubectl apply -f deployment/inference-service.yaml
	@echo "✅ Inference service deployed"

test-inference:
	@echo "Testing inference service..."
	python -m pytest tests/test_integration.py -v

run-api:
	@echo "Starting API server..."
	uvicorn autoshield.api.server:app --app-dir src --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf build/ dist/ *.egg-info
	find . -name "*.pyc" -delete

generate-data:
	@echo "Generating training dataset..."
	python src/scripts/generate_training_data.py
	@echo "✅ Dataset generated"

explore-data:
	@echo "Starting data exploration notebook..."
	jupyter notebook notebooks/01-data-exploration.ipynb

test-dataset:
	@echo "Testing dataset quality..."
	python src/scripts/run_test_dataset.py -v

	
train-model:
	@echo "Training CNN-LSTM model..."
	python src/scripts/train_model.py

evaluate-model:
	@echo "Evaluating model..."
	python src/scripts/evaluate_model.py

optimize-model:
	@echo "Optimizing model for inference..."
	python src/scripts/optimize_model.py  # We'll create this next

benchmark-latency:
	@echo "Benchmarking inference latency..."
	python src/scripts/benchmark_latency.py