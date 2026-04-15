RAG_ANSWER_PROMPT_TEMPLATE = (
    "Answer the user's question using only the retrieved Oracle knowledge base context. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}"
)
