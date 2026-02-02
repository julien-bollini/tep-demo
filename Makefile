VENV        := venv
PYTHON      := $(VENV)/bin/python3
PIP         := $(VENV)/bin/pip
DOCKER      := docker compose
OS          := $(shell uname -s)
FORCE       ?= false

# Flag management for local pipeline execution
ifeq ($(FORCE),true)
    FORCE_FLAG := --force
endif

.PHONY: help setup pipeline up dev down clean

# --- NOTIFICATIONS (Standardized for macOS) ---
define notify
  @if [ "$(OS)" = "Darwin" ]; then \
    osascript -e 'display notification "$(1) complete! 🚀" with title "MLOps"'; \
    say "Task $(1) is complete"; \
  fi
endef

help: ## Display this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = \":.*?## \"}; {printf \"\033[36m%-15s\033[0m %s\n\", $$1, $$2}'

setup: ## 1. [LOCAL] Create venv and install dependencies
	@echo "🌐 Provisioning local environment..."
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Setup finished."
	$(call notify,Setup)

pipeline: ## 2. [NATIVE] ETL + Training (Use FORCE=true to bypass cache)
	@echo "🧠 Running native ML pipeline (Force=$(FORCE))..."
	$(PYTHON) main.py $(FORCE_FLAG)
	@echo "✅ Artifacts generated in /models and /data"
	$(call notify,Pipeline)

up: ## 3. [DOCKER] Start API and Dashboard services (Fast start)
	@echo "🚀 Launching containerized services..."
	$(DOCKER) up -d api dashboard
	@echo "🎉 Dashboard: http://localhost:8501 | API: http://localhost:8000"

dev: ## 4. [DOCKER] Rebuild and restart services (Hot-reload)
	@echo "🔄 Synchronizing service updates..."
	$(DOCKER) up -d --build api dashboard
	$(call notify,Docker Services)

down: ## Stop all active Docker containers
	@echo "🛑 Tearing down services..."
	$(DOCKER) down

clean: ## Deep clean: Remove venv and pycache
	@echo "🧹 Cleaning workspace..."
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
