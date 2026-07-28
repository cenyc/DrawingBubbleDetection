#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "$script_dir/.." && pwd)"
compose_file="$script_dir/compose.yml"
release_file="$script_dir/release.env"
current_file="$script_dir/current-image-tag"
previous_file="$script_dir/previous-image-tag"

if [[ $# -ne 1 || -z "${1// }" ]]; then
  echo "Usage: $0 <image-tag>" >&2
  exit 2
fi

image_tag="$1"
if [[ ! "$image_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid image tag: $image_tag" >&2
  exit 2
fi
if [[ ! -f "$root_dir/.env" ]]; then
  echo "Missing $root_dir/.env; copy .env.example and set a strong BUBBLE_API_KEY." >&2
  exit 1
fi

current_tag=""
if [[ -f "$current_file" ]]; then
  current_tag="$(tr -d '\r\n' < "$current_file")"
fi

tmp_release="$release_file.tmp"
printf 'IMAGE_TAG=%s\n' "$image_tag" > "$tmp_release"
mv "$tmp_release" "$release_file"

compose=(
  docker compose
  --project-directory "$root_dir"
  --env-file "$root_dir/.env"
  --env-file "$release_file"
  -f "$compose_file"
)

echo "Pulling BubbleIQ images tagged $image_tag ..."
if ! "${compose[@]}" pull; then
  echo "Image pull failed." >&2
  if [[ -n "$current_tag" ]]; then
    printf 'IMAGE_TAG=%s\n' "$current_tag" > "$release_file"
  fi
  exit 1
fi

echo "Starting BubbleIQ ..."
if ! "${compose[@]}" up -d --remove-orphans --wait --wait-timeout 240; then
  echo "Deployment health check failed." >&2
  if [[ -n "$current_tag" ]]; then
    echo "Restoring previous tag $current_tag ..." >&2
    printf 'IMAGE_TAG=%s\n' "$current_tag" > "$release_file"
    "${compose[@]}" up -d --remove-orphans --wait --wait-timeout 240
  fi
  exit 1
fi

if [[ -n "$current_tag" && "$current_tag" != "$image_tag" ]]; then
  printf '%s\n' "$current_tag" > "$previous_file"
fi
printf '%s\n' "$image_tag" > "$current_file"

echo "Deployment successful: $image_tag"
"${compose[@]}" ps
