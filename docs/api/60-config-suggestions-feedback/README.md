# Config, Suggestions, and Feedback

## GET `/api/config`

Returns frontend/runtime config values such as:

- region
- embed model id
- model list
- model display names
- collection list
- feedback enablement

## POST `/api/suggestions`

Generate 3–6 follow-up questions from the last assistant message.

### Example body

```json
{
  "last_message": "Oracle vector search combines embeddings with database-native retrieval.",
  "model": null
}
```

### Response

```json
{
  "suggestions": [
    "How does hybrid search work?",
    "What models are supported?"
  ]
}
```

## POST `/api/feedback`

Submit a rating for a question/answer pair.

### Example body

```json
{
  "question": "What is Oracle vector search?",
  "answer": "It is ...",
  "feedback": 5
}
```

### Notes

This endpoint can return:

- `403` if feedback is disabled
- `503` if feedback service is unavailable
- `400` for invalid rating/input
