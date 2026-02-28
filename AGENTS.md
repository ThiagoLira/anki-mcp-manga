# Agents Guide

## Deployment

### Server

Connection details are in `.env` (gitignored):
- `DEPLOY_HOST` — server IP for SSH
- `ssh root@$DEPLOY_HOST` (uses default SSH key)
- **Project path**: `/opt/anki_mcp`

### Deploy

After pushing changes to `main`:

```bash
ssh root@$DEPLOY_HOST "cd /opt/anki_mcp && git pull && docker compose up --build -d"
```

### Containers

- `anki-bot` — Telegram bot (aiogram + LangGraph agent). No exposed ports (uses Telegram polling). Uses ONNX panel detector (~166MB) instead of PyTorch (~2GB).
- `anki-sync` — Anki sync server, bound to Tailscale IP only (not publicly accessible).

### Networking

The sync server is only reachable via Tailscale:
- Hostname and bind IP are in `.env` (`TAILSCALE_HOSTNAME`, `SYNC_BIND_IP`)
- Port 8080 is bound to the Tailscale interface, not the public IP
- Anki clients sync via `http://$TAILSCALE_HOSTNAME:8080`

### Panel detection model

The ONNX model (`models/panel_detector.onnx` + `.data`, ~166MB) is gitignored. After cloning, either:
- SCP from an existing machine: `scp -r models/ root@$DEPLOY_HOST:/opt/anki_mcp/models/`
- Re-export: `python scripts/export_panel_onnx.py` (requires `pip install ".[panels-torch]"`)

### Data

Persistent data lives in `/opt/anki_mcp/data/` (gitignored):
- `data/mcp/` — Bot's Anki collection and media
- `data/sync/` — Sync server data for Anki clients
