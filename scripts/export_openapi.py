#!/usr/bin/env python3
"""
Export OpenAPI JSON from api.main:app for regression testing.

Deterministically serializes the OpenAPI spec with sorted keys and consistent formatting.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from api.main import app


def main():
    # Get the OpenAPI spec
    openapi_spec = app.openapi()

    # Deterministically serialize
    output = json.dumps(openapi_spec, sort_keys=True, indent=2, ensure_ascii=False)

    # Write to stdout for piping, or to file if specified
    if len(sys.argv) > 1:
        output_file = Path(sys.argv[1])
        _ = output_file.write_text(output, encoding="utf-8")
        print(f"OpenAPI spec exported to {output_file}")
    else:
        print(output)


if __name__ == "__main__":
    main()
