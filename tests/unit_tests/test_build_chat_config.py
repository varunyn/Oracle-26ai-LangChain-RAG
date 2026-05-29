from unittest.mock import MagicMock, patch

import pytest

from api.dependencies import build_chat_config


class TestBuildChatConfig:
    """Unit tests for build_chat_config function."""

    @pytest.fixture(autouse=True)
    def reset_warning_flag(self):
        """Reset the module-level warning flag before each test."""
        import api.dependencies

        api.dependencies._warned_about_mcp_server_keys = False

    @patch("api.dependencies.get_settings")
    @patch("api.dependencies.get_mcp_servers_config")
    def test_mode_explicit_wins(self, mock_mcp_config, mock_settings):
        """Test that explicit mode parameter wins over all defaults."""
        mock_settings.return_value = MagicMock()
        mock_mcp_config.return_value = {"server1": {}}

        result = build_chat_config(mode="rag")

        assert result["configurable"]["mode"] == "rag"

    @patch("api.dependencies.get_settings")
    @patch("api.dependencies.get_mcp_servers_config")
    def test_default_mixed_when_mcp_enabled_and_config_nonempty(
        self, mock_mcp_config, mock_settings
    ):
        """Test default mode is 'mixed' when MCP tools are enabled and servers are configured."""
        mock_settings.return_value.ENABLE_MCP_TOOLS = True
        mock_mcp_config.return_value = {"server1": {"url": "http://example.com"}}

        result = build_chat_config()

        assert result["configurable"]["mode"] == "mixed"

    @patch("api.dependencies.get_settings")
    @patch("api.dependencies.get_mcp_servers_config")
    def test_default_rag_when_mcp_disabled(self, mock_mcp_config, mock_settings):
        """Test default mode is 'rag' when ENABLE_MCP_TOOLS=False."""
        mock_settings.return_value.ENABLE_MCP_TOOLS = False
        mock_mcp_config.return_value = {"server1": {"url": "http://example.com"}}

        result = build_chat_config()

        assert result["configurable"]["mode"] == "rag"

    @patch("api.dependencies.get_settings")
    @patch("api.dependencies.get_mcp_servers_config")
    def test_default_rag_when_config_empty(self, mock_mcp_config, mock_settings):
        """Test default mode is 'rag' when no MCP servers are configured."""
        mock_settings.return_value.ENABLE_MCP_TOOLS = True
        mock_mcp_config.return_value = {}

        result = build_chat_config()

        assert result["configurable"]["mode"] == "rag"

    @patch("api.dependencies.get_settings")
    @patch("api.dependencies.get_mcp_servers_config")
    def test_default_rag_when_config_none(self, mock_mcp_config, mock_settings):
        """Test default mode is 'rag' when MCP server config is None."""
        mock_settings.return_value.ENABLE_MCP_TOOLS = True
        mock_mcp_config.return_value = None

        result = build_chat_config()

        assert result["configurable"]["mode"] == "rag"

    @patch("api.dependencies.get_settings")
    @patch("api.dependencies.get_mcp_servers_config")
    @patch("api.dependencies.logger")
    def test_warning_logged_once_for_mcp_server_keys_param(
        self, mock_logger, mock_mcp_config, mock_settings
    ):
        """Test warning is logged once when mcp_server_keys parameter is provided."""
        mock_settings.return_value = MagicMock()
        mock_mcp_config.return_value = {}

        # First call should log warning
        build_chat_config(mcp_server_keys=["server1"])
        mock_logger.warning.assert_called_once_with(
            "MCP_SERVER_KEYS/mcp_server_keys does not choose the default mode. Mode is determined by ENABLE_MCP_TOOLS and the configured MCP server list, while MCP_SERVER_KEYS still limits which configured MCP servers/tools are loaded."
        )

        # Reset mock
        mock_logger.reset_mock()

        # Second call should not log again
        build_chat_config(mcp_server_keys=["server2"])
        mock_logger.warning.assert_not_called()

    @patch("api.dependencies.get_settings")
    @patch("api.dependencies.get_mcp_servers_config")
    @patch("api.dependencies.logger")
    def test_warning_logged_once_for_mcp_server_keys_setting(
        self, mock_logger, mock_mcp_config, mock_settings
    ):
        """Test warning is logged once when MCP_SERVER_KEYS setting is provided."""
        mock_settings.return_value.MCP_SERVER_KEYS = ["server1"]
        mock_mcp_config.return_value = {}

        # First call should log warning
        build_chat_config()
        mock_logger.warning.assert_called_once_with(
            "MCP_SERVER_KEYS/mcp_server_keys does not choose the default mode. Mode is determined by ENABLE_MCP_TOOLS and the configured MCP server list, while MCP_SERVER_KEYS still limits which configured MCP servers/tools are loaded."
        )

        # Reset mock
        mock_logger.reset_mock()

        # Second call should not log again
        build_chat_config()
        mock_logger.warning.assert_not_called()

    @patch("api.dependencies.get_settings")
    @patch("api.dependencies.get_mcp_servers_config")
    def test_mcp_server_keys_are_still_included_in_run_config(self, mock_mcp_config, mock_settings):
        """Test mcp_server_keys still limit which configured MCP servers are loaded."""
        mock_settings.return_value = MagicMock(
            LLM_MODEL_ID="test-model",
            EMBED_MODEL_TYPE="test-embed",
            ENABLE_RERANKER=True,
            DEFAULT_COLLECTION="test-collection",
            MCP_MAX_ROUNDS=3,
            ENABLE_MCP_TOOLS=True,
            RAG_SEARCH_MODE="vector",
            MCP_SERVER_KEYS=None,
        )
        mock_mcp_config.return_value = {"server1": {"url": "http://example.com"}}

        result = build_chat_config(mcp_server_keys=["server1"])

        assert result["configurable"]["mcp_server_keys"] == ["server1"]

    @patch("api.dependencies.get_settings")
    @patch("api.dependencies.get_mcp_servers_config")
    def test_default_mcp_server_key_is_explicit_in_run_config(self, mock_mcp_config, mock_settings):
        mock_settings.return_value = MagicMock(
            LLM_MODEL_ID="test-model",
            EMBED_MODEL_TYPE="test-embed",
            ENABLE_RERANKER=True,
            DEFAULT_COLLECTION="test-collection",
            MCP_MAX_ROUNDS=3,
            ENABLE_MCP_TOOLS=True,
            RAG_SEARCH_MODE="vector",
            MCP_SERVER_KEYS=None,
        )
        mock_mcp_config.return_value = {
            "default": {"url": "http://default.example.com/mcp"},
            "secondary": {"url": "http://secondary.example.com/mcp"},
        }

        result = build_chat_config()

        assert result["configurable"]["mcp_server_keys"] == ["default"]
        assert "mcp_url" not in result["configurable"]

    @patch("api.dependencies.get_settings")
    @patch("api.dependencies.get_mcp_servers_config")
    def test_run_config_shape_unchanged(self, mock_mcp_config, mock_settings):
        """Test that run_config output shape is unchanged."""
        mock_settings.return_value = MagicMock(
            LLM_MODEL_ID="test-model",
            EMBED_MODEL_TYPE="test-embed",
            ENABLE_RERANKER=True,
            DEFAULT_COLLECTION="test-collection",
            MCP_MAX_ROUNDS=3,
            ENABLE_MCP_TOOLS=False,
        )
        mock_mcp_config.return_value = {}

        result = build_chat_config(
            model_id="custom-model", thread_id="custom-thread", collection_name="custom-collection"
        )

        expected_keys = {"configurable"}
        assert set(result.keys()) == expected_keys

        configurable = result["configurable"]
        expected_config_keys = {
            "model_id",
            "embed_model_type",
            "search_mode",
            "enable_reranker",
            "enable_tracing",
            "collection_name",
            "thread_id",
            "mode",
            "max_rounds",
        }
        assert set(configurable.keys()) == expected_config_keys

        assert configurable["model_id"] == "custom-model"
        assert configurable["thread_id"] == "custom-thread"
        assert configurable["collection_name"] == "custom-collection"
        assert configurable["search_mode"] == mock_settings.return_value.RAG_SEARCH_MODE
        assert configurable["mode"] == "rag"
