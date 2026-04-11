#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p videos outputs

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose is required. Install Docker with the 'docker compose' plugin." >&2
  exit 1
fi

echo "Starting sheep carcass counter with Docker Compose..."
echo "Dashboard: http://127.0.0.1:8000/"

docker compose up --build sheep-counter
