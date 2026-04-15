# Generated Endpoint Reference

This file is generated from FastAPI OpenAPI via `scripts/sync_api_docs.py`.
Do not edit manually.

## GET `/api/config`

- operationId: `get_config_api_config_get`
- tags: config
- summary: Get Config

## DELETE `/api/documents/source`

- operationId: `delete_document_source_api_documents_source_delete`
- tags: documents
- summary: Delete Document Source

## GET `/api/documents/sources`

- operationId: `list_document_sources_api_documents_sources_get`
- tags: documents
- summary: List Document Sources

## POST `/api/documents/upload`

- operationId: `upload_documents_api_documents_upload_post`
- tags: documents
- summary: Upload Documents

## POST `/api/feedback`

- operationId: `post_feedback_api_feedback_post`
- tags: feedback
- summary: Post Feedback

## POST `/api/langgraph/threads`

- operationId: `create_thread_api_langgraph_threads_post`
- tags: langgraph-runtime
- summary: Create Thread

## POST `/api/langgraph/threads/{thread_id}/history`

- operationId: `get_thread_history_api_langgraph_threads__thread_id__history_post`
- tags: langgraph-runtime
- summary: Get Thread History

## POST `/api/langgraph/threads/{thread_id}/runs`

- operationId: `run_thread_api_langgraph_threads__thread_id__runs_post`
- tags: langgraph-runtime
- summary: Run Thread

## POST `/api/langgraph/threads/{thread_id}/runs/stream`

- operationId: `stream_thread_run_api_langgraph_threads__thread_id__runs_stream_post`
- tags: langgraph-runtime
- summary: Stream Thread Run

## GET `/api/langgraph/threads/{thread_id}/state`

- operationId: `get_thread_state_api_langgraph_threads__thread_id__state_get`
- tags: langgraph-runtime
- summary: Get Thread State

## POST `/api/suggestions`

- operationId: `post_suggestions_api_suggestions_post`
- tags: suggestions
- summary: Post Suggestions

## DELETE `/api/threads/{thread_id}`

- operationId: `delete_thread_api_threads__thread_id__delete`
- tags: langgraph-runtime
- summary: Delete Thread

## GET `/health`

- operationId: `health_health_get`
- tags: health
- summary: Health
