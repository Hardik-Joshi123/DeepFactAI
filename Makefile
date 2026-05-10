.PHONY: help install dev backend frontend test clean docs

help:
	@echo "Fake News Detection System - Development Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install         Install all dependencies"
	@echo "  make setup          Initialize project structure"
	@echo ""
	@echo "Development:"
	@echo "  make dev            Start development servers (backend + frontend)"
	@echo "  make backend        Start backend server only"
	@echo "  make frontend       Start frontend dev server only"
	@echo ""
	@echo "Training:"
	@echo "  make train-lstm     Train LSTM model"
	@echo "  make train-bert     Train BERT model"
	@echo "  make evaluate       Evaluate all models"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make lint           Run linting"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          Clean build artifacts"
	@echo "  make docs           Generate documentation"
	@echo "  make docker         Build Docker image"

install:
	cd backend && pip install -e .
	cd frontend && npm install

setup:
	python scripts/setup_models.py
	cp .env.example .env.local

dev:
	vercel dev

backend:
	cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

train-lstm:
	python scripts/train_lstm.py

train-bert:
	python scripts/train_bert.py

evaluate:
	python scripts/evaluate_models.py

test:
	pytest backend/tests/

lint:
	cd backend && ruff check .
	cd frontend && npm run lint

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	cd frontend && rm -rf .next
	rm -rf build dist *.egg-info

docs:
	@echo "Documentation setup"

docker:
	docker build -t fake-news-detector:latest .

docker-run:
	docker run -p 3000:3000 -p 8000:8000 fake-news-detector:latest
