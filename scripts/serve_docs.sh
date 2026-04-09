#!/usr/bin/env bash
# Serve the docs/ folder so you can view the Docsify site locally.
# Open http://localhost:3333 in your browser.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "Serving docs at http://localhost:3333 (Ctrl+C to stop)"
exec python3 -m http.server 3333 --directory docs
