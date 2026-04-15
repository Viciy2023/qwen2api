#!/bin/sh
set -eu

mkdir -p /data

init_json_file() {
  file_path="$1"
  default_content="$2"

  if [ ! -f "$file_path" ]; then
    printf '%s\n' "$default_content" > "$file_path"
  fi
}

init_json_file "${ACCOUNTS_FILE:-/data/accounts.json}" '[]'
init_json_file "${USERS_FILE:-/data/users.json}" '[]'
init_json_file "${CAPTURES_FILE:-/data/captures.json}" '[]'
init_json_file "${CONFIG_FILE:-/data/config.json}" '{}'

exec "$@"
