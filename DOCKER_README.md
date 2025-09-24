# AI Product Price Checker - Docker Setup

This project mirrors the deployment approach used in `pdf-auditor`: a single FastAPI container fronted by an optional Cloudflare tunnel for external access. Local development stays simple with hot reload and an exposed port.

## Prerequisites

1. **Docker** and **Docker Compose v2**
2. **Required credentials** in `.env`:
   - `OPENROUTER_API_KEY`
   - `FIREWORKS_API_KEY` (for OCR via Qwen2.5-VL)
   - `OXYLABS_WEB_API`
   - `OXYLABS_PROXY`
   - `CLOUDFLARE_TUNNEL_TOKEN` (only when using the tunnel)

Copy the sample file and populate your secrets:

```bash
cp env.example .env
# edit .env and fill in the values
```

## Running Locally (Development Profile)

```bash
# Rebuild and run with live reload and host port 8000 exposed
docker compose --profile development up --build
```

Access `http://localhost:8000` for the web UI. Code, data, screenshots, and uploads are mounted from the host for easy iteration.

## Production With Cloudflare Tunnel

1. Ensure your Cloudflare tunnel token is in `.env` as `CLOUDFLARE_TUNNEL_TOKEN`.
2. Start the production stack (no host ports exposed):

```bash
docker compose --profile production up -d --build
```

The `cloudflared` sidecar will automatically register the tunnel and proxy traffic to the internal `price-checker` service on port 8000. Manage access controls from Cloudflare Zero Trust as needed.

## Useful Commands

```bash
# View logs for the app container
docker compose logs -f price-checker

# View logs for the tunnel container
docker compose --profile production logs -f cloudflared

# Stop and remove containers, networks, and volumes
docker compose down
```

## Notes

- The `price-checker` service does **not** expose ports in production mode; all ingress flows through Cloudflare.
- Persistent data directories (`data`, `screenshots`, `uploads`) are backed by named Docker volumes when running production profile.
- A lightweight `/healthz` endpoint exists for the container health check.
- OCR is handled by Fireworks AI’s Qwen2.5-VL model; no local Tesseract/EasyOCR needed. 