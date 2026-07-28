# BubbleIQ Ubuntu deployment

The production deployment runs the Next.js frontend and the Python detector in
separate containers. Only the frontend is bound to the host, on
`127.0.0.1:3100`; the detector remains private inside the Compose network.

## 1. Prepare the server

Install Docker from Ubuntu packages:

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
```

Clone this repository into `/opt/bubble-platform`, then create the private
runtime configuration:

```bash
cd /opt/bubble-platform
cp .env.example .env
cp deploy/release.env.example deploy/release.env
openssl rand -hex 32
```

Put the generated value in `.env` as `BUBBLE_API_KEY`, then protect the file:

```bash
chmod 600 .env
```

For private GHCR packages, authenticate once with a GitHub token that has
`read:packages` permission:

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u cenyc --password-stdin
```

Do not store `GHCR_TOKEN` in this repository.

## 2. Configure Nginx

Create a dedicated domain in BT Panel and reverse proxy it to
`http://127.0.0.1:3100`. The required timeout and upload settings are shown in
`deploy/nginx-bubble.conf.example`. Enable HTTPS through BT Panel after DNS is
pointing at the server.

Neither port 3100 nor port 8001 should be opened in UFW. Port 3100 is bound only
to loopback and port 8001 is not published by Compose.

## 3. Deploy and update

GitHub Actions publishes `latest`, `sha-<commit>`, and Git tag images. Deploy an
immutable version:

```bash
cd /opt/bubble-platform
git pull --ff-only origin master
./deploy/deploy.sh sha-0123456
```

For a tagged release:

```bash
./deploy/deploy.sh v1.0.0
```

Inspect status and logs:

```bash
docker compose \
  --env-file .env \
  --env-file deploy/release.env \
  -f deploy/compose.yml ps

docker compose \
  --env-file .env \
  --env-file deploy/release.env \
  -f deploy/compose.yml logs -f --tail=200
```

Rollback to the previously successful image tag:

```bash
./deploy/rollback.sh
```

## 4. Local container build

Docker Compose can also build both images directly from the repository:

```bash
cp .env.example .env
# Replace the placeholder BUBBLE_API_KEY before continuing.
docker compose \
  --env-file .env \
  --env-file deploy/release.env.example \
  -f deploy/compose.yml build
```

The detector intentionally runs one Uvicorn worker because it holds mutable
per-request detector state. Scale up only after that state becomes
request-scoped.
