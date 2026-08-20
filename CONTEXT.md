# Domain Context

## Tool-agent execution

- **ToolAgentTurn**: The per-request prepared input for MCP and mixed-mode tool-agent execution: question, chat history, selected model, run configuration, prompt, available tools, and optional Oracle retrieval evidence. It is transient execution data, not a persisted chat-history value.
- **ToolExecutionTranscript**: The normalized record of a completed tool-agent execution: tool invocations and results, tools used, the terminal agent answer, and execution failures. It is derived from LangChain messages and consumed by mode-specific outcome policy.
