# Anki Telegram Bot

Telegram bot that creates Japanese vocabulary Anki flashcards from manga screenshots and text. Send a photo of a manga page and the bot extracts vocabulary, creates cards with the image, and syncs to all your devices.

Powered by a LangGraph ReAct agent (via OpenRouter) and a headless Anki collection running in Docker.

## Card Types

**Kanji cards** (deck: `Japones KANJI`):
- Front: kanji or word
- Back: reading (hiragana) + meaning

**Manga vocab cards** (deck: `Japones Vocab Mangas`):
- Front: manga panel screenshot + Japanese sentence with the target word in **bold**
- Back: full sentence translation with the target word in **bold**

## Architecture

```
Telegram User
    | photo + caption / text
    v
+-----------------------------+
|  anki-bot container         |
|                             |
|  aiogram (polling) ---------+--> Telegram API (outbound)
|      |                      |
|  LangGraph ReAct Agent -----+--> OpenRouter API (outbound)
|      |                      |
|  Anki Tools (direct calls)  |
|      |                      |
|  AnkiManager + SyncManager -+--> anki-sync:8080 (internal)
|      |                      |
|  /data (collection + media) |
+-----------------------------+

+-----------------------------+
|  anki-sync container        |
|  python -m anki.syncserver  |
|  :8080 (Tailscale for       |
|   phone/desktop clients)    |
+-----------------------------+
```

No inbound ports on the bot container. It uses Telegram polling (outbound only). The sync server is exposed to Tailscale for your phone/desktop Anki clients.

## Setup

### Prerequisites

- Docker (or Podman)
- A Telegram bot token (from [@BotFather](https://t.me/BotFather))
- An [OpenRouter](https://openrouter.ai/) API key

### 1. Clone and configure

```bash
git clone https://github.com/ThiagoLira/anki-mcp-manga.git
cd anki-mcp-manga

cp .env.example .env
```

Edit `.env` and fill in the required values:

```bash
# Required: get this from @BotFather on Telegram
TELEGRAM_BOT_TOKEN=your-bot-token-here

# Required: get this from openrouter.ai/keys
OPENROUTER_API_KEY=your-openrouter-key-here
```

#### Getting a Telegram bot token

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot`, pick a name and username
3. Copy the token it gives you into `TELEGRAM_BOT_TOKEN`

#### Restricting access (optional)

To limit who can use the bot, set `ALLOWED_TELEGRAM_USER_IDS` to a comma-separated list of Telegram user IDs. Leave empty to allow anyone.

To find your user ID, message [@userinfobot](https://t.me/userinfobot) on Telegram.

```bash
ALLOWED_TELEGRAM_USER_IDS=123456789,987654321
```

### 2. Start the stack

```bash
docker compose up --build -d
```

This starts two containers:
- **anki-bot** -- the Telegram bot (no exposed ports)
- **anki-sync** (:8080) -- self-hosted Anki sync server

Check the logs to confirm the bot started:

```bash
docker compose logs -f anki-bot
```

### 3. Talk to the bot

Open your bot in Telegram and try:

- `/start` -- welcome message
- `/stats` -- collection statistics
- `/decks` -- list all decks
- Send a text message like `"Create a card for 食べる (たべる, to eat)"` -- creates a kanji card and syncs
- Send a manga screenshot -- the agent extracts vocabulary, creates manga vocab cards with the image, and syncs

### 4. Connect Anki clients to the sync server

On your phone or desktop Anki (must be on the same network, or use Tailscale):

1. **Preferences > Syncing > Self-hosted sync server**
2. URL: `http://<your-server-ip>:8080`
3. Username / password: from your `.env` (`SYNC_USER` / `SYNC_PASSWORD`, default `user` / `password`)

If you use Tailscale, the URL would be `http://100.x.x.x:8080` or `http://<machine>.<tailnet>.ts.net:8080`.

## Data Management

### Where data lives

```
data/
  mcp/                    <-- bot's Anki collection
    collection.anki2
    collection.media/     <-- images (manga panels as .webp)
  sync/                   <-- sync server data
    user/
```

The `data/` directory is gitignored. This is the only state you need to back up.

### Import an existing collection

```bash
docker compose stop anki-bot

cp ~/.local/share/Anki2/User\ 1/collection.anki2 data/mcp/
cp -a ~/.local/share/Anki2/User\ 1/collection.media/. data/mcp/collection.media/

docker compose start anki-bot
```

Then send any message to the bot so it triggers a sync, or restart the stack.

### Migration back to AnkiWeb

1. Open Anki desktop
2. Remove the custom sync server URL from preferences
3. Sync to AnkiWeb -- choose "Upload to AnkiWeb"

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | yes | -- | Bot token from @BotFather |
| `OPENROUTER_API_KEY` | yes | -- | OpenRouter API key |
| `OPENROUTER_MODEL` | no | `anthropic/claude-sonnet-4` | LLM model to use |
| `ALLOWED_TELEGRAM_USER_IDS` | no | (empty = allow all) | Comma-separated Telegram user IDs |
| `SYNC_USER` | no | `user` | Sync server username |
| `SYNC_PASSWORD` | no | `password` | Sync server password |
| `SYNC_ENDPOINT` | no | `http://anki-sync:8080` | Sync server URL (internal) |
| `COLLECTION_PATH` | no | `/data/collection.anki2` | Collection path inside container |
| `KANJI_DECK` | no | `Japones KANJI` | Deck name for kanji cards |
| `MANGA_DECK` | no | `Japones Vocab Mangas` | Deck name for manga vocab cards |
| `SYNC_USER1` | no | `user:password` | Sync server credentials (`user:pass` format) |

## Development

```bash
# Create venv (Python 3.11 required by anki package)
uv venv --python 3.11 .venv
source .venv/bin/activate
uv sync --extra dev

# Run tests
TELEGRAM_BOT_TOKEN=test OPENROUTER_API_KEY=test pytest tests/ -v

# Run bot locally (without Docker)
source .env
python -m src.bot
```
