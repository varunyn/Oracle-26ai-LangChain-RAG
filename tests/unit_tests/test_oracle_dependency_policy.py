from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_oracle_code_uses_langchain_oracledb_distance_strategy() -> None:
    oracle_sources = [
        PROJECT_ROOT / "src" / "rag_agent" / "ingestion.py",
        PROJECT_ROOT / "src" / "rag_agent" / "infrastructure" / "oci_models.py",
        PROJECT_ROOT / "scripts" / "create_rag_table.py",
    ]

    for source_path in oracle_sources:
        source = source_path.read_text(encoding="utf-8")
        assert "from langchain_oracledb.vectorstores.utils import DistanceStrategy" in source
        assert "langchain_community.vectorstores.utils" not in source


def test_langchain_community_is_not_a_direct_dependency() -> None:
    pyproject_source = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"langchain-community' not in pyproject_source
