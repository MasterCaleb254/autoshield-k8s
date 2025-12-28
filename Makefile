.PHONY: help install test build deploy clean build-inference deploy-inference test-inference run-api generate-data explore-data test-dataset train-model evaluate-model optimize-model benchmark-latency deploy-policy-engine deploy-all test-pipeline run-demo monitor-actions rollback-action build-dashboard deploy-dashboard deploy-monitoring open-dashboard view-metrics test-dashboard deploy-complete

help:
	@echo "AutoShield-K8s Commands:"
	@echo "  install     Install dependencies"
	@echo "  test        Run all tests"
	@echo "  build       Build Docker images"
	@echo "  build-inference  Build inference service image"
	@echo "  deploy      Deploy to Kubernetes"
	@echo "  deploy-inference Deploy inference service to Kubernetes"
	@echo "  deploy-policy-engine Deploy policy engine/orchestrator to Kubernetes"
	@echo "  deploy-all  Deploy inference + policy engine/orchestrator + dashboard
  build-dashboard  Build dashboard Docker image
  deploy-dashboard  Deploy dashboard to Kubernetes
  deploy-monitoring  Deploy monitoring stack (Prometheus)
  open-dashboard  Open dashboard in browser
  view-metrics  View Prometheus metrics
  test-dashboard  Test dashboard health
  deploy-complete  Deploy all components (inference, policy, dashboard, monitoring)"
	@echo "  test-inference   Run inference integration tests"
	@echo "  test-pipeline    Run end-to-end pipeline test"
	@echo "  run-api     Start FastAPI inference API server"
	@echo "  run-demo    Run demo scenario"
	@echo "  monitor     Start monitoring stack"
	@echo "  monitor-actions  Tail orchestrator logs and filter executed actions"
	@echo "  rollback-action  Roll back the last action (script required)"
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

deploy-policy-engine:
	@echo "Deploying policy engine/orchestrator..."
	bash src/scripts/deploy-policy-engine.sh
	@echo "✅ Policy engine/orchestrator deployed"

# Dashboard and Monitoring
build-dashboard:
	@echo "Building dashboard..."
	docker build -f docker/dashboard.Dockerfile -t autoshield/dashboard:latest .

# Deployment targets
deploy-dashboard:
	@echo "Deploying dashboard..."
	kubectl apply -f deployment/dashboard.yaml
	@echo "✅ Dashboard deployed"

deploy-monitoring:
	@echo "Deploying monitoring stack..."
	kubectl apply -f monitoring/prometheus/prometheus.yaml
	@echo "✅ Monitoring stack deployed"

deploy-all:
	@echo "Deploying AutoShield core components..."
	make deploy-inference
	make deploy-policy-engine
	@echo "✅ Core components deployed"

deploy-complete:
	@echo "Deploying complete AutoShield system..."
	make deploy-inference
	make deploy-policy-engine
	make deploy-dashboard
	make deploy-monitoring
	@echo "✅ All components deployed"
	@echo "Dashboard: http://localhost:8081"
	@echo "Metrics: http://localhost:9090/metrics"

# Dashboard utilities
open-dashboard:
	@echo "Opening dashboard..."
	@if command -v xdg-open > /dev/null; then \
		xdg-open http://localhost:8081; \
	elif command -v open > /dev/null; then \
		open http://localhost:8081; \
	else \
		echo "Open http://localhost:8081 in your browser"; \
	fi

view-metrics:
	@echo "Viewing metrics..."
	@curl -s http://localhost:9090/metrics | head -50

test-dashboard:
	@echo "Testing dashboard..."
	@if curl -s -f http://localhost:8081/health > /dev/null; then \
		echo "✅ Dashboard healthy"; \
	else \
		echo "❌ Dashboard unhealthy"; \
		exit 1; \
	fi

test-inference:
	@echo "Testing inference service..."
	python -m pytest tests/test_integration.py -v

test-pipeline:
	@echo "Testing complete pipeline..."
	python -m pytest tests/test_end_to_end.py -v

run-demo:
	@echo "Running demo scenario..."
	@echo "Missing script: src/scripts/run_demo.py"
	@exit 1

monitor-actions:
	@echo "Monitoring recent actions..."
	kubectl logs -l component=orchestrator -n autoshield-system --tail=50 | grep "Action executed"

rollback-action:
	@echo "Rolling back last action..."
	@echo "Missing script: src/scripts/rollback_last_action.py"
	@exit 1

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