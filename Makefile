.PHONY: help up down status stacks-up stacks-down stacks-status core-up core-down core-logs observability-up observability-down observability-status langfuse-up langfuse-down langfuse-status

help:
	@echo "Targets:"
	@echo "  up/down/status                     - Run core + auto stacks from .env flags"
	@echo "  core-up/core-down/core-logs        - Run backend+frontend with docker compose (no Python)"
	@echo "  observability-up/down/status       - Manage observability stack via scripts/manage_stacks.py"
	@echo "  langfuse-up/down/status            - Manage langfuse stack via scripts/manage_stacks.py"
	@echo "  stacks-up/down/status              - Manage all enabled stacks via scripts/manage_stacks.py"

# Run core + auto-selected stacks based on .env flags and DOCKER_STACKS
up:
	$(MAKE) core-up
	uv run python scripts/manage_stacks.py up

down:
	uv run python scripts/manage_stacks.py down
	$(MAKE) core-down

status:
	docker compose ps
	uv run python scripts/manage_stacks.py status

# Core app (backend + frontend) using compose directly
core-up:
	docker compose up -d backend frontend

core-down:
	docker compose down

core-logs:
	docker compose logs -f backend

# Observability stack (preferred via manage_stacks.py)
observability-up:
	uv run python scripts/manage_stacks.py up --stacks observability

observability-down:
	uv run python scripts/manage_stacks.py down --stacks observability

observability-status:
	uv run python scripts/manage_stacks.py status --stacks observability

# Langfuse stack (preferred via manage_stacks.py)
langfuse-up:
	uv run python scripts/manage_stacks.py up --stacks langfuse

langfuse-down:
	uv run python scripts/manage_stacks.py down --stacks langfuse

langfuse-status:
	uv run python scripts/manage_stacks.py status --stacks langfuse

# All enabled/auto-included stacks
stacks-up:
	uv run python scripts/manage_stacks.py up

stacks-down:
	uv run python scripts/manage_stacks.py down

stacks-status:
	uv run python scripts/manage_stacks.py status
