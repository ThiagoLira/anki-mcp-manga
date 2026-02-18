# Anki MCP

Headless Anki MCP server that runs in Docker, lets Claude Web create Japanese vocabulary flashcards (from manga panels), and syncs them to all devices via a self-hosted Anki sync server.

## Architecture

### Network Topology

How requests flow from Claude Web to your Anki devices:

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

How the Python modules import each other:

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

### Card Creation Flows

#### Kanji card — `create_kanji_card`

```mermaid
sequenceDiagram
    participant C as Claude Web
    participant A as BearerAuthMiddleware
    participant M as FastMCP Server
    participant AM as AnkiManager
    participant NT as note_templates
    participant COL as Collection (.anki2)

    C->>A: POST /mcp (Bearer token)
    A->>A: Validate token
    A->>M: Forward request

    M->>AM: create_kanji_card(<br/>kanji, reading, meaning, tags)

    AM->>COL: Lazy open Collection(path)
    AM->>NT: ensure_kanji_notetype(col)
    NT->>COL: models.by_name("Kanji")

    alt Notetype doesn't exist
        NT->>COL: models.new() + add 3 fields<br/>(Kanji, Reading, Meaning)
        NT->>COL: add Card 1 template
        NT->>COL: models.add(notetype)
    end

    NT-->>AM: notetype dict

    AM->>COL: decks.id("Japones KANJI")
    AM->>COL: new_note(notetype)
    AM->>AM: Set Kanji, Reading, Meaning

    opt tags provided
        AM->>AM: note.add_tag() for each
    end

    AM->>COL: col.add_note(note, deck_id)
    AM-->>M: CardResult(note_id, front, deck)
    M-->>C: {"status": "created", ...}
```

#### Manga card — `create_manga_card`

```mermaid
sequenceDiagram
    participant C as Claude Web
    participant A as BearerAuthMiddleware
    participant M as FastMCP Server
    participant AM as AnkiManager
    participant NT as note_templates
    participant COL as Collection (.anki2)
    participant MEDIA as Media Folder

    C->>A: POST /mcp (Bearer token)
    A->>A: Validate token
    A->>M: Forward request

    M->>AM: create_manga_card(<br/>word, translation,<br/>image_base64, tags)

    AM->>COL: Lazy open Collection(path)
    AM->>NT: ensure_manga_notetype(col)
    NT->>COL: models.by_name("Manga Vocab")

    alt Notetype doesn't exist
        NT->>COL: models.new() + add 3 fields<br/>(Word, Image, Translation)
        NT->>COL: add Card 1 template
        NT->>COL: models.add(notetype)
    end

    NT-->>AM: notetype dict

    AM->>COL: decks.id("Japones Vocab Mangas")
    AM->>COL: new_note(notetype)
    AM->>AM: Set Word, Translation

    opt image_base64 provided
        AM->>AM: base64 decode → Pillow open
        AM->>AM: Convert RGB, resize max 1024px
        AM->>AM: Compress to WebP quality=80
        AM->>MEDIA: col.media.write_data(<br/>anki_mcp_{hash}.webp)
        AM->>AM: Image = &lt;img src="file.webp"&gt;
    end

    opt tags provided
        AM->>AM: note.add_tag() for each
    end

    AM->>COL: col.add_note(note, deck_id)
    AM-->>M: CardResult(note_id, front, deck)
    M-->>C: {"status": "created", ...}
```

### Sync Flow

What happens when Claude calls `sync_to_server`:

```mermaid
sequenceDiagram
    participant C as Claude Web
    participant SM as SyncManager
    participant COL as Collection
    participant SS as Sync Server (:8080)

    C->>SM: sync_to_server()

    SM->>COL: sync_login(user, pass, endpoint)
    COL->>SS: Authenticate
    SS-->>COL: SyncAuth (hkey + endpoint)

    SM->>COL: sync_collection(auth, sync_media=False)
    COL->>SS: Check sync state
    SS-->>COL: SyncCollectionResponse(required)

    alt required == NO_CHANGES (0)
        SM->>SM: result = "no_changes"
    else required == NORMAL_SYNC (1)
        SM->>SM: result = "synced"
    else required == FULL_UPLOAD (4)
        SM->>COL: full_upload_or_download(upload=True)
        COL->>SS: Push entire collection
        SM->>COL: reopen(after_full_sync=True)
    else required == FULL_DOWNLOAD (3)
        SM->>COL: full_upload_or_download(upload=False)
        COL->>SS: Pull entire collection
        SM->>COL: reopen(after_full_sync=True)
    else required == FULL_SYNC (2)
        SM->>COL: full_upload_or_download(upload=True)
        Note over SM: Ambiguous state:<br/>default to upload<br/>since MCP is the source
        SM->>COL: reopen(after_full_sync=True)
    end

    SM->>COL: sync_login() again<br/>(re-auth after potential reopen)
    SM->>COL: sync_media(auth)
    COL->>SS: Push/pull media files (images)

    SM-->>C: {"collection_sync": "...", "media_sync": "synced"}
```

### Card Types

The two notetypes and the cards they produce:

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

## Setup

```bash
cp .env.example .env
# Edit .env with your values
docker compose up -d
```

### Tailscale Funnel

```bash
tailscale funnel --bg 8000
```

### Claude Web Connector

Settings > Connectors > Add custom connector:
- URL: `https://<machine>.ts.net/mcp`
- Auth: Bearer token from `.env`
