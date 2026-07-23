#!/bin/bash
# This file contains bash commands that will be executed at the end of the container build process,
# after all system packages and programming language specific package have been installed.
#
# Note: This file may be removed if you don't need to use it

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python3 -m pip install --no-cache-dir --upgrade pip

if [ -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    cd "${PROJECT_ROOT}"
    python3 -m pip install --no-cache-dir -e .[dev]
else
    python3 -m pip install --no-cache-dir \
        "fastapi>=0.111" \
        "httpx>=0.27" \
        "langgraph>=0.2" \
        "qdrant-client>=1.9" \
        "redis>=5.0" \
        "typedb-driver>=3.0" \
        "uvicorn[standard]>=0.30" \
        "pytest>=8.2" \
        "pytest-asyncio>=0.23" \
        "respx>=0.21" \
        "anyio[trio]>=4.4"
fi
