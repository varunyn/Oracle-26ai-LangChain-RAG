# Graph

## GET `/graph/mermaid`

Returns the LangGraph workflow as Mermaid text.

### Example

```bash
curl -s http://127.0.0.1:3002/graph/mermaid
```

### Use cases

- architecture docs
- debugging route topology
- visualizing workflow behavior

### Notes

The response is plain text, not JSON.
