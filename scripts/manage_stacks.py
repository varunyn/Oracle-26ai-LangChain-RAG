#!/usr/bin/env python3
"""Helper to start/stop docker stacks defined in DOCKER_STACKS (from .env / api/settings)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_stacks() -> dict[str, dict[str, object]]:
    from api.settings import get_settings

    stacks = get_settings().DOCKER_STACKS
    if not isinstance(stacks, dict):
        sys.exit("DOCKER_STACKS must be a dict (set in .env or api/settings defaults)")
    return cast(dict[str, dict[str, object]], stacks)


def _available_stacks() -> str:
    stacks = _load_stacks()
    return ", ".join(sorted(stacks)) or "<none>"


def _require_docker() -> None:
    if shutil.which("docker") is None:
        sys.exit(
            "Docker is required but was not found in PATH. Install Docker Desktop and try again."
        )


def _ensure_files(name: str, stack: Mapping[str, object], action: str) -> Path:
    compose_file = Path(cast(str, stack.get("compose_file", ""))).expanduser()
    compose_path = (ROOT / compose_file).resolve()
    if not compose_file or not compose_path.exists():
        sys.exit(f"Stack '{name}' compose file not found: {compose_file}")

    if action == "up":
        env_entry = stack.get("env_file")
        if env_entry:
            env_files: Sequence[str]
            if isinstance(env_entry, (list, tuple)):
                env_files = cast(Sequence[str], env_entry)
            else:
                env_files = [cast(str, env_entry)]
            for env in env_files:
                env_path = (ROOT / Path(env).expanduser()).resolve()
                if not env_path.exists():
                    sys.exit(
                        f"Stack '{name}' requires env file '{env}'. Copy the example file and fill secrets before starting."
                    )
    return compose_path


def _docker_command(name: str, stack: Mapping[str, object], action: str) -> list[str]:
    compose_path = _ensure_files(name, stack, action)
    cmd: list[str] = ["docker", "compose", "-f", str(compose_path)]

    profiles: Sequence[str] = cast(Sequence[str], stack.get("profiles") or [])
    for profile in profiles:
        cmd.extend(["--profile", profile])

    services: Sequence[str] | None = cast(Sequence[str] | None, stack.get("services"))
    if action == "status":
        cmd.append("ps")
    elif action == "up":
        cmd.extend(["up", "-d"])
        if services:
            cmd.extend(list(services))
    elif action == "down":
        cmd.append("down")
    else:
        raise ValueError(f"Unsupported action: {action}")
    return cmd


def _run_stack(name: str, stack: Mapping[str, object], action: str) -> None:
    cmd = _docker_command(name, stack, action)
    print(f"[{name}] $ {' '.join(cmd)}")
    try:
        _ = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        msg = (
            f"[{name}] docker compose command failed with exit code {exc.returncode}. "
            "See output above for details."
        )
        print(msg, file=sys.stderr)
        sys.exit(exc.returncode)


def main() -> None:
    from api.settings import get_settings

    parser = argparse.ArgumentParser(
        description="Manage docker compose stacks defined in config.DOCKER_STACKS",
    )
    _ = parser.add_argument(
        "action",
        choices=["up", "down", "status"],
        help="Which docker action to run",
    )
    _ = parser.add_argument(
        "stacks_pos",
        nargs="*",
        metavar="STACK",
        help=(
            "Stack names to operate on (default: all stacks with enabled=True). "
            "Examples: 'up core', 'status observability', 'down langfuse'"
        ),
    )
    _ = parser.add_argument(
        "--stacks",
        nargs="+",
        metavar="STACK",
        help="[Deprecated] Prefer positional STACK arguments. Specific stacks to operate on.",
    )
    args = parser.parse_args()

    settings = get_settings()
    stacks = _load_stacks()
    if not stacks:
        sys.exit("DOCKER_STACKS is empty. Configure in .env or see api/settings.py defaults.")

    # Determine target stack names (positional preferred, but also honor --stacks)
    target_names: list[str] = []
    if getattr(args, "stacks_pos", None):
        target_names.extend(cast(Sequence[str], args.stacks_pos))
    if getattr(args, "stacks", None):
        for n in cast(Sequence[str], args.stacks):
            if n not in target_names:
                target_names.append(n)

    if not target_names:
        # Fallback to enabled=True stacks
        target_names = [name for name, data in stacks.items() if data.get("enabled")]

        if settings.ENABLE_OBSERVABILITY_STACK or settings.ENABLE_OTEL_TRACING:
            target_names.append("observability")
        if settings.ENABLE_LANGFUSE_TRACING:
            target_names.append("langfuse")

        if target_names:
            target_names = sorted(set(target_names))
        else:
            sys.exit(
                "No stacks selected. Provide STACK names positionally (e.g., 'up core') or set enabled=True in DOCKER_STACKS (.env / api/settings)."
            )

    # Validate stack names
    for name in target_names:
        if name not in stacks:
            sys.exit(f"Unknown stack '{name}'. Available: {_available_stacks()}")

    _require_docker()

    for name in target_names:
        _run_stack(name, stacks[name], cast(str, args.action))


if __name__ == "__main__":
    main()
