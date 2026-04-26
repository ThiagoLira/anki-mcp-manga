# Agents Guide

## Deployment

> **READ THIS FIRST.** Production is **k3s**, not docker compose. The repo
> ships a `docker-compose.yml` but it exists only so the deploy script can
> build images locally; **never run `docker compose up` on the deploy host
> as a way to start the bot** — it brings up a parallel instance that
> fights the k3s pod for the Telegram polling slot (Telegram returns
> `TelegramConflictError: terminated by other getUpdates request`).
>
> **There is exactly one way to deploy: `bash k8s/deploy.sh`.**

### Server

- **Host**: `teagolab-1` (Tailscale MagicDNS hostname)
- **SSH user**: `thiago` (i.e. `ssh thiago@teagolab-1`)
- **Project path**: `~/repos/anki_mcp` on the server (note: dir is `anki_mcp`, not `anki-mcp-manga`)
- **Deployment style**: k3s, namespace `anki`. Manifests in `k8s/anki.yaml`.
- **Pods**: `anki-bot-*` and `anki-sync-*` in the `anki` namespace.

### Deploy

After pushing changes to `main`:

```bash
ssh thiago@teagolab-1 "cd ~/repos/anki_mcp && git pull && bash k8s/deploy.sh"
```

`k8s/deploy.sh` does the whole rollout:
1. `docker compose build` — builds the images locally on the host
2. `docker save | sudo k3s ctr images import` — loads them into k3s's containerd
3. `kubectl apply -f k8s/anki.yaml` — applies any manifest changes
4. `kubectl rollout restart deployment/anki-bot deployment/anki-sync -n anki` — restarts the pods on the new image
5. `kubectl rollout status` — waits for the rollout

The script needs sudo for the `k3s ctr images import` step. Run it where you can type the password (or where thiago has passwordless sudo).

### If you accidentally `docker compose up`

Two bot processes will be polling the same Telegram token and one will be
losing on every `getUpdates`. Symptom in `kubectl logs deployment/anki-bot -n anki`
or `docker compose logs anki-bot`:

```
TelegramConflictError: terminated by other getUpdates request
```

Fix:

```bash
ssh thiago@teagolab-1 "cd ~/repos/anki_mcp && docker compose down"
```

The k3s pod will recover within ~10s once the conflicting docker container is gone.

### Containers

- `anki-bot` — Telegram bot (aiogram + structured LLM calls). No exposed ports (uses Telegram polling). Uses ONNX panel detector (~166MB) instead of PyTorch (~2GB). Pulls candidate words via the deterministic pipeline (manga-ocr + fugashi + wordfreq); only the per-page batched translation hits the LLM.
- `anki-sync` — Anki sync server, exposed via a k3s `LoadBalancer` Service (`anki-sync-external`) bound to the Tailscale IP only.

### Networking

The sync server is only reachable via Tailscale:
- Service `anki-sync-external` is a k3s `LoadBalancer` that binds to the Tailscale IP only (not the public IP).
- Anki clients sync via `http://teagolab-1:8080` (or the equivalent Tailscale IP).

### LLM backend

`CardAgent.__init__` probes `local_llm_url` (default `http://janus:8080/v1` over Tailscale, see `src/config.py`) at startup and uses it if reachable; otherwise falls back to OpenRouter (`OPENROUTER_API_KEY`). The local backend on `janus` is a `llama-server` instance — bind it to `0.0.0.0` (not `127.0.0.1`) for the tailnet probe to reach it.

### Panel detection model

The ONNX model (`models/panel_detector.onnx` + `.data`, ~166MB) is gitignored. After cloning to a new deploy host, either:
- SCP from an existing machine: `scp -r models/ thiago@teagolab-1:~/repos/anki_mcp/models/`
- Re-export: `python scripts/export_panel_onnx.py` (requires `pip install ".[panels-torch]"`)

### Data

Persistent data lives in `~/repos/anki_mcp/data/` (gitignored), mounted into the pods as hostPath volumes (see `k8s/anki.yaml`):
- `data/mcp/` — Bot's Anki collection and media
- `data/sync/` — Sync server data for Anki clients
