# Anki MCP

Headless Anki MCP server that runs in Docker, lets Claude create Japanese vocabulary flashcards (from manga panels), and syncs them to all devices via a self-hosted Anki sync server.

## Card Types

Two straightforward card types matching standard Anki study patterns:

**Kanji cards** (deck: `Japones KANJI`) — for kanji and vocabulary:
- Front: the kanji or word
- Back: reading (hiragana) + meaning

**Manga vocab cards** (deck: `Japones Vocab Mangas`) — for vocabulary from manga:
- Front: the Japanese word + a screenshot of the manga panel
- Back: translation of the full sentence from the panel

## Quick Start (Local with Claude Code)

### 1. Clone and configure

```bash
git clone https://github.com/ThiagoLira/anki-mcp-manga.git
cd anki-mcp-manga

cp .env.example .env
# Edit .env — at minimum set MCP_AUTH_TOKEN to something strong:
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Start the stack

```bash
# Docker
docker compose up --build -d

# Or Podman
podman compose up --build -d
```

This starts two containers:
- **anki-mcp** (:8000) — the MCP server
- **anki-sync** (:8080) — the Anki sync server

### 3. Connect Claude Code

Create `.mcp.json` in the project root (gitignored):

```json
{
  "mcpServers": {
    "anki": {
      "type": "http",
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN_HERE"
      }
    }
  }
}
```

Start a new Claude Code session in this directory. It will connect to the MCP server and have access to all the Anki tools.

### 4. Test it

Ask Claude Code to:
- `"list my anki decks"`
- `"create a kanji card for 食べる (たべる, to eat)"`
- `"sync to server"`

## Expose to Claude Web (Tailscale Funnel)

To use this from Claude Web (claude.ai), you need to expose the MCP server over HTTPS. Tailscale Funnel does this with automatic TLS certificates.

### 1. Install Tailscale

See [tailscale.com/download](https://tailscale.com/download).

### 2. Enable Funnel

```bash
sudo tailscale funnel 8000
```

This maps `https://<machine>.<tailnet>.ts.net/` to `localhost:8000`.

### 3. Add connector in Claude Web

Go to [claude.ai/settings/integrations](https://claude.ai/settings/integrations):
1. Click **Add more integrations** → **Custom integration**
2. **Name**: `Anki`
3. **URL**: `https://<machine>.<tailnet>.ts.net/mcp`
4. **Authentication**: Bearer Token → paste your `MCP_AUTH_TOKEN`

Claude Web can now create flashcards directly in conversation.

### 4. Point Anki clients at the sync server

On your phone/desktop Anki (must be on the same Tailscale network):
- **Preferences → Syncing → Self-hosted sync server**
- URL: `http://<machine>.<tailnet>.ts.net:8080` or `http://100.x.x.x:8080`
- Username/password: from your `.env` (`SYNC_USER` / `SYNC_PASSWORD`)

## Data Management

### Where data lives

```
data/
├── mcp/                    ← MCP server's Anki collection
│   ├── collection.anki2
│   └── collection.media/   ← images (manga panels, etc.)
└── sync/                   ← Sync server's data
    └── user/
```

The `data/` directory is gitignored. This is the only state you need to back up.

### Import your existing Anki collection

If you already have an Anki collection you want to use:

```bash
# Stop the MCP container first
docker compose stop anki-mcp

# Copy your collection
cp ~/.local/share/Anki2/User\ 1/collection.anki2 data/mcp/
cp -a ~/.local/share/Anki2/User\ 1/collection.media/. data/mcp/collection.media/

# Restart
docker compose start anki-mcp
```

Then call `sync_to_server` from Claude to push it to the sync server.

### Sync workflow

```
Claude creates card → MCP server adds to local collection
                    → call sync_to_server
                    → pushes to self-hosted sync server
                    → phone/desktop Anki syncs from same server
```

After creating cards, always call `sync_to_server` to push changes. When Anki desktop syncs, if there's a conflict, choose **"Download from server"** to get the MCP's cards.

### Backup

Just back up the `data/` directory. Or rely on the sync server — any Anki client pointed at it has a full copy.

### Migration back to AnkiWeb

1. Open Anki desktop
2. Remove the custom sync server URL from preferences
3. Sync to AnkiWeb — choose **"Upload to AnkiWeb"**

## Architecture

### Network Topology

```mermaid
graph LR
    subgraph Internet
        CW[Claude Web]
    end

    subgraph Tailscale Funnel
        TF["HTTPS endpoint<br/>&lt;machine&gt;.ts.net/mcp"]
    end

    subgraph Docker Compose
        subgraph anki-mcp ["anki-mcp container :8000"]
            AUTH[BearerAuthMiddleware]
            MCP[FastMCP Server]
            AM[AnkiManager]
            SM[SyncManager]
            COL[(collection.anki2)]
            MEDIA[(collection.media/)]
        end

        subgraph anki-sync ["anki-sync container :8080"]
            SS[anki.syncserver]
            SD[(sync data)]
        end
    end

    subgraph Tailscale Mesh
        PHONE[Phone Anki]
        DESKTOP[Desktop Anki]
    end

    CW -->|"MCP over HTTP<br/>Bearer token"| TF
    TF --> AUTH
    AUTH --> MCP
    MCP --> AM
    MCP --> SM
    AM --> COL
    AM --> MEDIA
    SM -->|"sync_login<br/>sync_collection<br/>sync_media"| SS
    SS --> SD
    PHONE -->|sync| SS
    DESKTOP -->|sync| SS
```

### Module Dependency Graph

```mermaid
graph TD
    SERVER["server.py<br/><i>FastMCP entry point</i><br/><i>8 tool definitions</i>"]
    AUTH["auth.py<br/><i>BearerAuthMiddleware</i><br/><i>checks header + query param</i>"]
    AM["anki_manager.py<br/><i>AnkiManager class</i><br/><i>create_kanji_card, create_manga_card</i>"]
    SM["sync_manager.py<br/><i>SyncManager class</i><br/><i>sync orchestration</i>"]
    NT["note_templates.py<br/><i>Kanji + Manga Vocab notetypes</i><br/><i>fields, CSS, card templates</i>"]
    CFG["config.py<br/><i>Pydantic Settings</i><br/><i>reads .env</i>"]

    SERVER --> AUTH
    SERVER --> AM
    SERVER --> SM
    AM --> CFG
    AM --> NT
    SM --> CFG
    AUTH --> CFG
    NT --> ANKI["anki (pip)<br/><i>Collection, models,</i><br/><i>decks, media, sync</i>"]
    AM --> ANKI
    SM -.->|"TYPE_CHECKING only"| AM
```

### Card Creation Flow (Kanji)

```mermaid
sequenceDiagram
    participant C as Claude
    participant A as Auth Middleware
    participant M as FastMCP Server
    participant AM as AnkiManager
    participant COL as Collection (.anki2)

    C->>A: POST /mcp (Bearer token)
    A->>M: Forward request
    M->>AM: create_kanji_card(kanji, reading, meaning, tags)
    AM->>COL: ensure_kanji_notetype(col)
    AM->>COL: decks.id("Japones KANJI")
    AM->>COL: new_note → set fields → add_note
    AM-->>M: CardResult
    M-->>C: {"status": "created", ...}
```

### Card Creation Flow (Manga)

```mermaid
sequenceDiagram
    participant C as Claude
    participant A as Auth Middleware
    participant M as FastMCP Server
    participant AM as AnkiManager
    participant COL as Collection (.anki2)
    participant MEDIA as Media Folder

    C->>A: POST /mcp (Bearer token)
    A->>M: Forward request
    M->>AM: create_manga_card(word, translation, image_url, tags)
    AM->>COL: ensure_manga_notetype(col)
    AM->>COL: decks.id("Japones Vocab Mangas")

    opt image_url provided
        AM->>AM: Download image from URL
        AM->>AM: Pillow: resize, compress to WebP
        AM->>MEDIA: col.media.write_data(file.webp)
    end

    AM->>COL: new_note → set fields → add_note
    AM-->>M: CardResult
    M-->>C: {"status": "created", ...}
```

### Sync Flow

```mermaid
sequenceDiagram
    participant C as Claude
    participant SM as SyncManager
    participant COL as Collection
    participant SS as Sync Server (:8080)

    C->>SM: sync_to_server()
    SM->>COL: sync_login(user, pass, endpoint)
    COL->>SS: Authenticate
    SM->>COL: sync_collection(auth)
    COL->>SS: Check sync state

    alt NO_CHANGES
        SM->>SM: Already up to date
    else NORMAL_SYNC
        SM->>SM: Incremental sync done
    else FULL_UPLOAD / FULL_SYNC
        SM->>COL: full_upload_or_download(upload=True)
        COL->>SS: Push entire collection
        SM->>SM: Close and reopen collection
    end

    SM->>COL: sync_media(auth)
    COL->>SS: Push/pull media files
    SM-->>C: {"collection_sync": "...", "media_sync": "synced"}
```

### Card Types

```mermaid
graph TD
    subgraph KANJI_NT ["Kanji Notetype → deck: Japones KANJI"]
        direction TB
        K1["Kanji<br/><code>準</code>"]
        K2["Reading<br/><code>じゅん</code>"]
        K3["Meaning<br/><code>level, conform</code>"]
    end

    subgraph KANJI_CARD ["Card"]
        direction TB
        KF["<b>Front</b><br/>準"]
        KB["<b>Back</b><br/>じゅん<br/>level, conform"]
        KF --- KB
    end

    KANJI_NT --> KANJI_CARD

    subgraph MANGA_NT ["Manga Vocab Notetype → deck: Japones Vocab Mangas"]
        direction TB
        M1["Word<br/><code>規則</code>"]
        M2["Image<br/><code>&lt;img src=panel.webp&gt;</code>"]
        M3["Translation<br/><code>You must follow the rules.</code>"]
    end

    subgraph MANGA_CARD ["Card"]
        direction TB
        MF["<b>Front</b><br/>規則<br/>+ manga panel screenshot"]
        MB["<b>Back</b><br/>You must follow the rules."]
        MF --- MB
    end

    MANGA_NT --> MANGA_CARD
```

### MCP Tools

```mermaid
graph LR
    subgraph TOOLS ["Available MCP Tools"]
        direction TB
        T1["<b>create_kanji_card</b><br/>kanji, reading, meaning → Japones KANJI"]
        T2["<b>create_manga_card</b><br/>word, image, translation → Japones Vocab Mangas"]
        T3["<b>create_kanji_cards_batch</b><br/>Multiple kanji cards at once"]
        T4["<b>create_manga_cards_batch</b><br/>Multiple manga cards at once"]
        T5["<b>sync_to_server</b><br/>Push changes to sync server"]
        T6["<b>list_decks</b><br/>All decks + note counts"]
        T7["<b>search_notes</b><br/>Anki search syntax, max 50"]
        T8["<b>get_collection_stats</b><br/>Notes, cards, decks, study"]
    end

    subgraph TARGET ["Backed by"]
        AM["AnkiManager"]
        SM["SyncManager"]
    end

    T1 --> AM
    T2 --> AM
    T3 --> AM
    T4 --> AM
    T5 --> SM
    T6 --> AM
    T7 --> AM
    T8 --> AM
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `MCP_AUTH_TOKEN` | Bearer token for MCP auth (required) | — |
| `SYNC_USER` | Sync server username | `user` |
| `SYNC_PASSWORD` | Sync server password | `password` |
| `SYNC_ENDPOINT` | Sync server URL (internal) | `http://anki-sync:8080` |
| `COLLECTION_PATH` | Path to collection inside container | `/data/collection.anki2` |
| `KANJI_DECK` | Target deck for kanji cards | `Japones KANJI` |
| `MANGA_DECK` | Target deck for manga cards | `Japones Vocab Mangas` |
| `SYNC_USER1` | Sync server credentials (`user:pass`) | `user:password` |

## Development

```bash
# Create venv with Python 3.11 (required by anki package)
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
MCP_AUTH_TOKEN=test pytest tests/ -v

# Run server locally (without Docker)
MCP_AUTH_TOKEN=test python -m src.server
```
