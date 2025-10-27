#!/usr/bin/env bash

set -Eeuo pipefail

current_user=$(whoami)
echo "Entering docker_entrypoint.sh as user:${current_user}..."

exec "$@"
