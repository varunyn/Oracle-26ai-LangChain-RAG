from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import httpx


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "setup_langfuse_project",
        Path("scripts/setup_langfuse_project.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def __init__(
        self,
        *,
        models: list[dict[str, Any]] | None = None,
        score_configs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.models = models or []
        self.score_configs = score_configs or []
        self.requests: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        self.requests.append((method, url, kwargs))
        request = httpx.Request(method, url)
        if method == "GET" and url.endswith("/api/public/models"):
            return httpx.Response(
                200,
                json={
                    "data": self.models,
                    "meta": {
                        "page": 1,
                        "limit": 100,
                        "totalItems": len(self.models),
                        "totalPages": 1,
                    },
                },
                request=request,
            )
        if method == "GET" and url.endswith("/api/public/score-configs"):
            return httpx.Response(
                200,
                json={
                    "data": self.score_configs,
                    "meta": {
                        "page": 1,
                        "limit": 100,
                        "totalItems": len(self.score_configs),
                        "totalPages": 1,
                    },
                },
                request=request,
            )
        if method == "POST" and url.endswith("/api/public/models"):
            body = dict(kwargs["json"])
            body.update({"id": "model-1", "isLangfuseManaged": False})
            self.models.append(body)
            return httpx.Response(200, json=body, request=request)
        if method == "POST" and url.endswith("/api/public/score-configs"):
            body = dict(kwargs["json"])
            body.update({"id": "score-config-1", "isArchived": False})
            self.score_configs.append(body)
            return httpx.Response(200, json=body, request=request)
        return httpx.Response(404, json={"message": "not found"}, request=request)

    def close(self) -> None:
        return None


def test_setup_creates_missing_model_and_score_config() -> None:
    module = load_module()
    client = FakeClient()
    setup = module.LangfuseProjectSetup(
        host="http://langfuse.local",
        public_key="pk-test",
        secret_key="sk-test",
        client=client,
    )

    model_result = setup.ensure_model(module.MODEL_DEFINITIONS[0])
    score_result = setup.ensure_score_config(module.SCORE_CONFIGS[0])

    assert model_result == "created model xai.grok-4.20-0309-reasoning"
    assert score_result == "created score config user-rating"
    assert any(
        method == "POST" and url.endswith("/api/public/models")
        for method, url, _ in client.requests
    )
    assert any(
        method == "POST" and url.endswith("/api/public/score-configs")
        for method, url, _ in client.requests
    )


def test_setup_is_idempotent_when_config_exists() -> None:
    module = load_module()
    model = module.MODEL_DEFINITIONS[0]
    score = module.SCORE_CONFIGS[0]
    client = FakeClient(
        models=[
            {
                "id": "model-1",
                "modelName": model.model_name,
                "matchPattern": model.match_pattern,
                "unit": model.unit,
                "inputPrice": float(model.input_price),
                "outputPrice": float(model.output_price),
                "isLangfuseManaged": False,
            }
        ],
        score_configs=[
            {
                "id": "score-config-1",
                "name": score.name,
                "dataType": score.data_type,
                "minValue": score.min_value,
                "maxValue": score.max_value,
                "isArchived": False,
            }
        ],
    )
    setup = module.LangfuseProjectSetup(
        host="http://langfuse.local",
        public_key="pk-test",
        secret_key="sk-test",
        client=client,
    )

    assert setup.ensure_model(model) == "exists model xai.grok-4.20-0309-reasoning"
    assert setup.ensure_score_config(score) == "exists score config user-rating"
    assert not any(method == "POST" for method, _, _ in client.requests)


def test_setup_reports_mismatched_existing_config_without_overwriting() -> None:
    module = load_module()
    model = module.MODEL_DEFINITIONS[0]
    score = module.SCORE_CONFIGS[0]
    client = FakeClient(
        models=[
            {
                "id": "model-1",
                "modelName": model.model_name,
                "matchPattern": model.match_pattern,
                "unit": model.unit,
                "inputPrice": 1,
                "outputPrice": float(model.output_price),
                "isLangfuseManaged": False,
            }
        ],
        score_configs=[
            {
                "id": "score-config-1",
                "name": score.name,
                "dataType": score.data_type,
                "minValue": 0,
                "maxValue": score.max_value,
                "isArchived": False,
            }
        ],
    )
    setup = module.LangfuseProjectSetup(
        host="http://langfuse.local",
        public_key="pk-test",
        secret_key="sk-test",
        client=client,
    )

    assert setup.ensure_model(model).startswith("mismatch model")
    assert setup.ensure_score_config(score).startswith("mismatch score config")
    assert not any(method == "POST" for method, _, _ in client.requests)
