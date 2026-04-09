import os
import sys

import pytest

# Ensure 'src' and project root are on sys.path (rag_agent package + api for integration tests)
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, os.pardir))
SRC = os.path.join(PROJECT_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def vcr_config():
    """VCR configuration for pytest-recording per LangChain docs.

    Filters sensitive headers and query parameters so recorded cassettes are safe to commit.
    """
    return {
        "filter_headers": [
            ("authorization", "XXXX"),
            ("x-api-key", "XXXX"),
            # OCI / OpenAI-compatible headers we may pass
            ("opc-request-id", "XXXX"),
            ("openai-organization", "XXXX"),
        ],
        "filter_query_parameters": [
            ("api_key", "XXXX"),
            ("key", "XXXX"),
            ("openai_api_key", "XXXX"),
        ],
    }


@pytest.fixture(scope="session", autouse=True)
def _set_langfuse_environment() -> None:
    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "development"
