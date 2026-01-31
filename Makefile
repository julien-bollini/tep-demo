# --- SETTINGS ---
VENV         := venv
VENV_PYTHON  := $(VENV)/bin/python3
VENV_PIP     := $(VENV)/bin/pip
DOCKER       := docker-compose
FORCE        ?= false
OS           := $(shell uname -s)

# --- FLAGS ---
ifeq ($(FORCE),true)
    FORCE_FLAG := --force
endif

.PHONY: help setup lint pipeline train evaluate deploy build up down clean clean-cache all

# --- NOTIFICATIONS ---
define notify
	@if [ "$(OS)" = "Darwin" ]; then \
		osascript -e 'display notification "$(1) terminé ! 🚀" with title "MLOps"'; \
		say "$(1) complete"; \
		afplay /System/Library/Sounds/Glass.aiff || afplay /System/Library/Sounds/Ping.aiff; \
	elif [ "$(OS)" = "Linux" ]; then \
		notify-send "MLOps" "$(1) terminé ! 🚀" || echo "🔔 Done"; \
		(command -v spd-say > /dev/null && spd-say "$(1) complete") || echo "\a"; \
	fi
endef

help: ## Display this help message with categorized commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- INSTALLATION & QUALITY ASSURANCE ---
all: setup lint ## Run full pipeline with FORCE=true to validate everything
	@echo "🛠️  Running full development validation..."
	$(VENV_PYTHON) main.py --step all --force
	$(call notify,Validation globale)

deploy: down build ## Build and start services using existing models (No Training)
	@echo "🚀 Orchestrating production deployment..."
	$(VENV_PYTHON) main.py --step evaluate
	$(DOCKER) up -d
	@echo "🎉 System online at http://localhost:8501"
	$(call notify,Déploiement)

setup: ## Bootstraps the local environment and synchronizes dependencies
	@echo "🌐 Provisioning virtual environment..."
	test -d $(VENV) || python3 -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@echo "✅ Setup complete."

lint: ## Static analysis and automated code formatting via Ruff
	@echo "🧐 Running Ruff for linting and security checks..."
	$(VENV_PYTHON) -m ruff check . --fix || true

# --- MLOPS WORKFLOW ---

pipeline: setup lint preprocess train evaluate ## End-to-end ML pipeline execution (ETL to Eval)
	@echo "⚙️ Executing full Machine Learning pipeline..."
	@if [ "$(FORCE)" = "true" ]; then \
		echo "🔥 FORCE mode detected: Invalidating existing model artifacts..."; \
		rm -f models/*.joblib; \
	fi
	@FORCE_REPROCESS=$(FORCE) $(VENV_PYTHON) main.py

preprocess: setup ## ETL: Data extraction, cleaning, and parquet serialization
	@echo "🧹 Triggering Preprocessing stage..."
	$(VENV_PYTHON) main.py --step preprocess

train: ## Model training: Executes cascaded learning logic
	@echo "🧠 Triggering Model Training..."
	@FORCE_REPROCESS=$(FORCE) $(VENV_PYTHON) main.py --step train

evaluate: ## Validation: Generates performance metrics and reports
	@echo "📊 Triggering Performance Evaluation..."
	@FORCE_REPROCESS=$(FORCE) $(VENV_PYTHON) main.py --step evaluate

# --- CONTAINER ORCHESTRATION ---

build: ## Build or rebuild Docker images defined in docker-compose.yml
	@echo "🏗️ Building container images..."
	$(DOCKER) build

up: ## Deploy services in detached mode (background)
	@echo "🚀 Orchestrating service startup..."
	$(DOCKER) up -d

down: ## Gracefully stop and remove containers, networks, and images
	@echo "🛑 Tearing down services..."
	$(DOCKER) down

restart: down up ## Atomic restart: Full teardown and redeployment

# --- SYSTEM MAINTENANCE ---

clean-cache: ## Prunes global Kagglehub cache to recover disk space
	@echo "🔥 Purging Kagglehub cache in ~/.cache/kagglehub..."
	rm -rf ~/.cache/kagglehub
	rm -f config/cache.yaml
	@echo "✅ Global cache cleared."

clean: down ## Full project reset: Removes venv, pycache, and build artifacts
	@echo "🧹 Deep cleaning project workspace..."
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f data/processed/**/*.parquet
	rm -f models/*.joblib
	rm -f config/cache.yaml
	@$(MAKE) clean-cache
	@echo "✨ Workspace is pristine."

# Force le recalcul des métriques ET la reconstruction Docker
force-deploy:
	@echo "🧪 Forcing metrics recomputation..."
	FORCE_REPROCESS=true python src/evaluation/evaluator.py
	@echo "🐳 Forcing Docker build..."
	docker-compose build --no-cache
	docker-compose up -d
