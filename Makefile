.PHONY: help install test build deploy clean

help:
	@echo "AutoShield-K8s Commands:"
	@echo "  install     Install dependencies"
	@echo "  test        Run all tests"
	@echo "  build       Build Docker images"
	@echo "  deploy      Deploy to Kubernetes"
	@echo "  monitor     Start monitoring stack"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ --cov=autoshield --cov-report=html

build:
	docker build -f docker/feature-extractor.Dockerfile -t autoshield/feature-extractor:latest .
	docker build -f docker/inference-service.Dockerfile -t autoshield/inference:latest .

deploy:
	kubectl apply -f deployment/

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
	python scripts/train_model.py

evaluate-model:
	@echo "Evaluating model..."
	python scripts/evaluate_model.py

optimize-model:
	@echo "Optimizing model for inference..."
	python scripts/optimize_model.py  # We'll create this next

benchmark-latency:
	@echo "Benchmarking inference latency..."
	python scripts/benchmark_latency.py