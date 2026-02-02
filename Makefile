# --- SETTINGS & INFRASTRUCTURE ---
VENV         := venv
VENV_PYTHON  := $(VENV)/bin/python3
VENV_PIP     := $(VENV)/bin/pip
DOCKER       := docker compose
FORCE        ?= false
OS           := $(shell uname -s)

# --- FLAGS ---
ifeq ($(FORCE),true)
    FORCE_FLAG := --force
endif

.PHONY: help setup lint pipeline build up down dev clean clean-cache install

# --- NOTIFICATIONS (Standardized for Linux/macOS) ---
define notify
  @if [ "$(OS)" = "Darwin" ]; then \
    osascript -e 'display notification "$(1) complete! 🚀" with title "MLOps"'; \
    afplay /System/Library/Sounds/Glass.aiff || afplay /System/Library/Sounds/Ping.aiff; \
  elif [ "$(OS)" = "Linux" ]; then \
    notify-send "MLOps" "$(1) complete! 🚀" || echo "🔔 Done"; \
    (command -v spd-say > /dev/null && spd-say "$(1) complete") || echo "\a"; \
  fi
endef

help: ## Display this help message with categorized commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- PROVISIONING & QUALITY ---
setup: ## Bootstraps the local environment and syncs dependencies
	@echo "🌐 Provisioning virtual environment..."
	test -d $(VENV) || python3 -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@$(call notify,Setup)

lint: ## Static analysis and automated code formatting via Ruff
	@echo "🧐 Running Ruff for linting and security checks..."
	$(VENV_PYTHON) -m ruff check . --fix || true

# --- SENIOR ORCHESTRATION ---
install: build ## [First Run] Build images and execute one-time training pipeline
	@echo "🏗️  Starting full installation and initial training..."
	$(DOCKER) run --rm pipeline
	@$(call notify,Full Installation)

up: ## [Daily Run] Launch production services (API & Dashboard)
	@echo "🚀 Launching containerized services..."
	$(DOCKER) up -d api dashboard
	@echo "🎉 System online at http://localhost:8501"
	@$(call notify,Deployment)

pipeline: ## [Manual] Force model retraining via Docker container
	@echo "🧠 Triggering Model Retraining..."
	docker compose run --rm pipeline python main.py --force
	@$(call notify,Pipeline Training)

dev: ## Fast update: Rebuild and restart services (Hot-reload)
	@echo "🔄 Synchronizing service updates..."
	$(DOCKER) up -d --build api dashboard

# --- DOCKER COMMANDS ---
build: ## Rebuild Docker images using multi-stage optimization
	@echo "🏗️ Building container images..."
	$(DOCKER) build

down: ## Stop and remove all active containers
	@echo "🛑 Tearing down services..."
	$(DOCKER) down

# --- SYSTEM MAINTENANCE & CLEANUP ---
clean-cache: ## Prunes global Kagglehub cache to recover disk space
	@echo "🔥 Purging Kagglehub cache..."
	rm -rf ~/.cache/kagglehub
	rm -f config/cache.yaml
	@echo "✅ Global cache cleared."

clean: down ## Deep clean: Removes venv, pycache, and artifacts
	@echo "🧹 Deep cleaning project workspace..."
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f data/processed/**/*.parquet
	rm -f models/*.joblib
	rm -f config/cache.yaml
	@$(MAKE) clean-cache
	@$(call notify,Clean Up)
