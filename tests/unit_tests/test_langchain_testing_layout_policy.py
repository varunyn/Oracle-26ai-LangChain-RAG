from pathlib import Path


def test_no_mocked_workflow_tests_use_integration_name_at_top_level() -> None:
    tests_root = Path(__file__).resolve().parents[1]
    top_level_integration_named = sorted(
        path.name
        for path in tests_root.glob("test_*integration*.py")
        if path.is_file()
    )

    assert top_level_integration_named == []


def test_langchain_test_directories_exist() -> None:
    tests_root = Path(__file__).resolve().parents[1]

    assert (tests_root / "unit_tests").is_dir()
    assert (tests_root / "integration_tests").is_dir()


def test_top_level_python_tests_are_reduced_to_uncategorized_remainder() -> None:
    tests_root = Path(__file__).resolve().parents[1]
    top_level_tests = sorted(
        path.name
        for path in tests_root.glob("test_*.py")
        if path.is_file()
    )

    assert top_level_tests == []


def test_unit_test_helpers_module_exists_for_fake_models() -> None:
    tests_root = Path(__file__).resolve().parents[1]

    assert (tests_root / "unit_tests" / "helpers.py").is_file()


def test_unit_test_helpers_module_supports_structured_output_adapters() -> None:
    tests_root = Path(__file__).resolve().parents[1]
    helper_source = (tests_root / "unit_tests" / "helpers.py").read_text(encoding="utf-8")

    assert "StructuredOutputFakeChatModel" in helper_source


def test_rewrite_unit_tests_no_longer_define_inline_structured_output_fake_llms() -> None:
    tests_root = Path(__file__).resolve().parents[1]
    rewrite_test_source = (
        tests_root / "unit_tests" / "test_rewrite_for_retrieval.py"
    ).read_text(encoding="utf-8")

    assert "class FakeStructuredModel" not in rewrite_test_source


def test_unit_test_helpers_module_supports_tool_call_message_builders() -> None:
    tests_root = Path(__file__).resolve().parents[1]
    helper_source = (tests_root / "unit_tests" / "helpers.py").read_text(encoding="utf-8")

    assert "tool_call_message" in helper_source


def test_pyproject_documents_categorized_test_layout() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pyproject_source = (project_root / "pyproject.toml").read_text(encoding="utf-8")

    assert "tests/unit_tests" in pyproject_source
    assert "tests/integration_tests" in pyproject_source
