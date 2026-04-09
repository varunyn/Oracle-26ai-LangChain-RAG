import os

import pytest
from langchain_core.messages import HumanMessage

from src.rag_agent.infrastructure.oci_models import get_llm


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("OCI_INTEGRATION_TESTS", "0") != "1",
    reason="Set OCI_INTEGRATION_TESTS=1 to run live LLM test",
)
def test_llm_invoke_returns_text():
    """
    Minimal live LLM call. VCR recording is disabled for OCI SDK due to
    incompatibility with vendored requests/urllib3. This runs live only.
    """
    llm = get_llm()
    result = llm.invoke([HumanMessage(content="Say 'hello' and nothing else.")])
    content = getattr(result, "content", None) or result
    assert isinstance(content, str)
    assert "hello" in content.lower()
