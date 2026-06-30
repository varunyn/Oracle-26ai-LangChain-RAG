# LangChain rule

For any LangChain-related task, do not answer from memory when current documentation can be checked.

We have mcp configure for docs-langchain use that

Always verify against the latest official LangChain/Langgraph docs before:

- writing code
- suggesting imports
- recommending APIs
- explaining agents, tools, models, memory, retrieval, or structured output
- proposing migrations or fixes

Prefer:

- official docs
- release notes
- migration guides

Do not rely on:

- old examples
- deprecated APIs

If docs cannot be checked, explicitly say the answer may be stale.
