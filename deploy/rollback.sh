#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
previous_file="$script_dir/previous-image-tag"

if [[ ! -s "$previous_file" ]]; then
  echo "No previous image tag has been recorded." >&2
  exit 1
fi

previous_tag="$(tr -d '\r\n' < "$previous_file")"
exec "$script_dir/deploy.sh" "$previous_tag"
