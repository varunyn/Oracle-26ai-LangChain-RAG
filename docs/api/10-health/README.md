# Health

## GET `/health`

Lightweight health check for load balancers and local verification.

### Example

```bash
curl -s http://127.0.0.1:3002/health
```

### Response

```json
{
  "status": "ok"
}
```

### Notes

This endpoint is intentionally lightweight and does not validate downstream dependencies such as database, OCI, or MCP connectivity.
