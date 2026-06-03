#!/usr/bin/env python3
"""Create project-level Langfuse config used by the local RAG app."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = ROOT / ".env"


@dataclass(frozen=True)
class ModelDefinition:
    model_name: str
    match_pattern: str
    unit: str
    input_price: Decimal
    output_price: Decimal
    source: str


@dataclass(frozen=True)
class ScoreConfig:
    name: str
    data_type: str
    min_value: int
    max_value: int
    description: str


MODEL_DEFINITIONS: tuple[ModelDefinition, ...] = (
    ModelDefinition(
        model_name="xai.grok-4.20-0309-reasoning",
        match_pattern=r"(?i)^xai\.grok-4\.20-0309-reasoning$",
        unit="TOKENS",
        input_price=Decimal("0.00000125"),
        output_price=Decimal("0.0000025"),
        source="xAI Grok 4.20 public API pricing: $1.25 input / $2.50 output per 1M tokens",
    ),
    ModelDefinition(
        model_name="meta.llama-4-scout-17b-16e-instruct",
        match_pattern=r"(?i)^meta\.llama-4-scout-17b-16e-instruct$",
        unit="TOKENS",
        input_price=Decimal("0.00000072"),
        output_price=Decimal("0.00000072"),
        source="OCI listing for Meta Llama 4 Scout: $0.720 input/output per 1M tokens",
    ),
)

SCORE_CONFIGS: tuple[ScoreConfig, ...] = (
    ScoreConfig(
        name="user-rating",
        data_type="NUMERIC",
        min_value=1,
        max_value=5,
        description=(
            "End-user star rating for a chat answer. Higher is better; values are submitted "
            "from the frontend feedback control."
        ),
    ),
)


def _decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _same_decimal(left: object, right: Decimal) -> bool:
    parsed = _decimal(left)
    return parsed is not None and parsed == right


class LangfuseProjectSetup:
    def __init__(
        self,
        *,
        host: str,
        public_key: str,
        secret_key: str,
        client: httpx.Client | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self._client = client or httpx.Client(
            auth=(public_key, secret_key),
            timeout=20,
        )

    def close(self) -> None:
        self._client.close()

    def _url(self, path: str) -> str:
        return f"{self.host}{path}"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self._client.request(method, self._url(path), **kwargs)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"Langfuse returned unexpected response for {path}")
        return data

    def list_models(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        page = 1
        while True:
            data = self._request("GET", "/api/public/models", params={"limit": 100, "page": page})
            items = data.get("data")
            if not isinstance(items, list):
                raise ValueError("Langfuse model list response missing data[]")
            models.extend(item for item in items if isinstance(item, dict))
            raw_meta = data.get("meta")
            meta = raw_meta if isinstance(raw_meta, dict) else {}
            total_pages = int(meta.get("totalPages") or page)
            if page >= total_pages or not items:
                return models
            page += 1

    def list_score_configs(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/public/score-configs")
        items = data.get("data")
        if not isinstance(items, list):
            raise ValueError("Langfuse score config response missing data[]")
        return [item for item in items if isinstance(item, dict)]

    def ensure_model(self, definition: ModelDefinition) -> str:
        existing = [
            model
            for model in self.list_models()
            if model.get("modelName") == definition.model_name
            and model.get("matchPattern") == definition.match_pattern
            and model.get("isLangfuseManaged") is False
        ]
        if existing:
            matching = [
                model
                for model in existing
                if model.get("unit") == definition.unit
                and _same_decimal(model.get("inputPrice"), definition.input_price)
                and _same_decimal(model.get("outputPrice"), definition.output_price)
            ]
            if matching:
                return f"exists model {definition.model_name}"
            return (
                f"mismatch model {definition.model_name}: custom definition exists with different "
                "pricing; review in Langfuse before changing it"
            )

        self._request(
            "POST",
            "/api/public/models",
            json={
                "modelName": definition.model_name,
                "matchPattern": definition.match_pattern,
                "unit": definition.unit,
                "inputPrice": float(definition.input_price),
                "outputPrice": float(definition.output_price),
            },
        )
        return f"created model {definition.model_name}"

    def ensure_score_config(self, config: ScoreConfig) -> str:
        existing = [item for item in self.list_score_configs() if item.get("name") == config.name]
        if existing:
            current = existing[0]
            matches = (
                current.get("isArchived") is False
                and current.get("dataType") == config.data_type
                and current.get("minValue") == config.min_value
                and current.get("maxValue") == config.max_value
            )
            if matches:
                return f"exists score config {config.name}"
            return (
                f"mismatch score config {config.name}: existing config has different shape; "
                "review in Langfuse before changing it"
            )

        self._request(
            "POST",
            "/api/public/score-configs",
            json={
                "name": config.name,
                "dataType": config.data_type,
                "minValue": config.min_value,
                "maxValue": config.max_value,
                "description": config.description,
            },
        )
        return f"created score config {config.name}"


def _env_value(name: str, value: str | None) -> str:
    if value and value.strip():
        return value.strip()
    from os import environ

    env_value = environ.get(name, "").strip()
    if env_value:
        return env_value
    raise SystemExit(f"Missing {name}. Set it in .env or pass the matching CLI flag.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Langfuse model pricing and score configs for this project.",
    )
    _ = parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to an env file containing LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, and LANGFUSE_SECRET_KEY.",
    )
    _ = parser.add_argument("--host", help="Override LANGFUSE_HOST.")
    _ = parser.add_argument("--public-key", help="Override LANGFUSE_PUBLIC_KEY.")
    _ = parser.add_argument("--secret-key", help="Override LANGFUSE_SECRET_KEY.")
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the configs this script manages without calling Langfuse.",
    )
    return parser.parse_args()


def _print_managed_configs() -> None:
    print("Managed Langfuse model pricing:")
    for definition in MODEL_DEFINITIONS:
        print(
            f"- {definition.model_name}: input={definition.input_price} "
            f"output={definition.output_price} unit={definition.unit}"
        )
    print("Managed Langfuse score configs:")
    for config in SCORE_CONFIGS:
        print(f"- {config.name}: {config.data_type} [{config.min_value}, {config.max_value}]")


def main() -> None:
    args = parse_args()
    env_file = Path(args.env_file).expanduser()
    if env_file.exists():
        load_dotenv(env_file)

    if args.dry_run:
        _print_managed_configs()
        return

    setup = LangfuseProjectSetup(
        host=_env_value("LANGFUSE_HOST", args.host),
        public_key=_env_value("LANGFUSE_PUBLIC_KEY", args.public_key),
        secret_key=_env_value("LANGFUSE_SECRET_KEY", args.secret_key),
    )
    try:
        for definition in MODEL_DEFINITIONS:
            print(setup.ensure_model(definition))
        for config in SCORE_CONFIGS:
            print(setup.ensure_score_config(config))
    finally:
        setup.close()


if __name__ == "__main__":
    try:
        main()
    except httpx.HTTPError as exc:
        sys.exit(f"Langfuse API request failed: {exc}")
