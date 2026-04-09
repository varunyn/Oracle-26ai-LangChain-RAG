# OCI Session Token (SECURITY_TOKEN) Setup

Use session tokens when you prefer short-lived credentials instead of long-lived API keys. Set `AUTH=SECURITY_TOKEN` in `.env`; the app then reads `security_token_file` and `key_file` from the profile you select via `OCI_PROFILE`.

## 1. Create a session token via OCI CLI

Let the CLI manage token files—do not edit `~/.oci/config` manually.

### Option A – Browser flow

```bash
oci session authenticate --profile-name CHICAGO --region us-chicago-1
```

- Choose your region when prompted, then sign in through the browser window.
- The CLI creates or updates the `[CHICAGO]` profile and writes `security_token_file` under `~/.oci/`.

### Option B – Headless / CI flow

Requires an existing API-key profile.

```bash
oci session authenticate --no-browser --profile CHICAGO --auth security_token
```

Optional shorter lifetime (default is 60 minutes):

```bash
oci session authenticate --no-browser --profile CHICAGO --auth security_token --session-expiration-in-minutes 30
```

## 2. Resulting profile

After authentication, `~/.oci/config` contains:

```ini
[CHICAGO]
user=ocid1.user.oc1..aaaaaaaaxxxxx
fingerprint=xx:xx:...
tenancy=ocid1.tenancy.oc1..aaaaaaaaxxxxx
region=us-chicago-1
key_file=/Users/you/.oci/your-private-key.pem
security_token_file=/Users/you/.oci/session_token_xxxxx
```

- `security_token_file` is added/updated automatically.
- `key_file` is still required to sign requests with the session token.

## 3. Configure this repo

```python
# .env
AUTH = "SECURITY_TOKEN"
OCI_PROFILE = "CHICAGO"  # profile defined in ~/.oci/config
```

Restart the FastAPI service (`./run_api.sh`). Logs will show the active auth mode, e.g., `auth=SECURITY_TOKEN`.

## 4. Refresh or validate tokens

- Refresh before expiry:

  ```bash
  oci session refresh --profile CHICAGO
  ```

- Validate an existing token:

  ```bash
  oci session validate --config-file ~/.oci/config --profile CHICAGO --auth security_token
  ```

Tokens expire after the duration set during authentication (default one hour). Re-run `authenticate` or `refresh` to keep sessions valid.

## 5. Troubleshooting

### Error: "Config value for 'security_token_file' must be specified when using --auth security_token"

The selected profile lacks a token entry.

1. Generate a token:
   - Browser: `oci session authenticate --profile-name YOUR_PROFILE --region us-chicago-1`
   - Headless: ensure `user`, `fingerprint`, `tenancy`, `region`, and `key_file` exist, then run `oci session authenticate --no-browser --profile YOUR_PROFILE --auth security_token`
2. Confirm `security_token_file=...` appears in `~/.oci/config` under `[YOUR_PROFILE]`.
3. In `.env`, set `AUTH=SECURITY_TOKEN` and `OCI_PROFILE=YOUR_PROFILE`; restart the backend.

If you intend to use API keys only, keep `AUTH = "API_KEY"` and skip the session-token CLI flags.

## References

- [Token-based Authentication for the CLI](https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/clitoken.htm)
- [SDK and CLI Configuration File](https://docs.oracle.com/iaas/Content/API/Concepts/sdkconfig.htm)
