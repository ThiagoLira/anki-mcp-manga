import atexit

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from .anki_manager import AnkiManager
from .config import settings
from .sync_manager import SyncManager

mcp = FastMCP(
    "Anki MCP",
    instructions="""\
MCP server for creating Japanese flashcards in Anki. There are two card types:

## Kanji cards (deck: Japones KANJI)
For learning kanji and vocabulary words. Straightforward:
- Front: the kanji or word
- Back: reading (hiragana) + meaning
Use create_kanji_card for these.

## Manga vocab cards (deck: Japones Vocab Mangas)
For vocabulary learned from manga panels. Straightforward:
- Front: the Japanese word + a screenshot of the manga panel
- Back: translation of the full sentence from the panel
Use create_manga_card for these. Always include the manga panel image when available.

After creating cards, call sync_to_server to push changes to the sync server.
""",
)

manager = AnkiManager()
sync = SyncManager(manager)
atexit.register(manager.close)


# --- Tools ---


@mcp.tool()
def create_kanji_card(
    kanji: str = Field(description="The kanji or Japanese word"),
    reading: str = Field(description="Hiragana reading"),
    meaning: str = Field(description="English meaning"),
    tags: list[str] | None = Field(
        default=None,
        description='Optional tags (e.g. ["N3", "verb"])',
    ),
) -> dict:
    """Create a kanji/vocab flashcard in the Japones KANJI deck.

    Front: kanji. Back: reading + meaning. Keep it simple.
    """
    result = manager.create_kanji_card(
        kanji=kanji, reading=reading, meaning=meaning, tags=tags
    )
    return {"status": "created", "note_id": result.note_id, "front": result.front}


@mcp.tool()
def create_manga_card(
    word: str = Field(description="The Japanese word from the manga panel"),
    translation: str = Field(description="Translation of the full sentence from the panel"),
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded screenshot of the manga panel",
    ),
    image_url: str | None = Field(
        default=None,
        description="URL of the manga panel image (server downloads it directly)",
    ),
    tags: list[str] | None = Field(
        default=None,
        description='Optional tags (e.g. ["manga-title", "chapter-1"])',
    ),
) -> dict:
    """Create a manga vocab flashcard in the Japones Vocab Mangas deck.

    Front: word + manga panel screenshot. Back: sentence translation.
    Provide image_url or image_base64 for the manga panel (URL preferred).
    """
    result = manager.create_manga_card(
        word=word, translation=translation,
        image_data=image_base64, image_url=image_url, tags=tags,
    )
    return {"status": "created", "note_id": result.note_id, "front": result.front}


class KanjiCardInput(BaseModel):
    kanji: str = Field(description="The kanji or Japanese word")
    reading: str = Field(description="Hiragana reading")
    meaning: str = Field(description="English meaning")
    tags: list[str] | None = None


class MangaCardInput(BaseModel):
    word: str = Field(description="The Japanese word from the manga panel")
    translation: str = Field(description="Translation of the full sentence")
    image_base64: str | None = None
    image_url: str | None = None
    tags: list[str] | None = None


@mcp.tool()
def create_kanji_cards_batch(cards: list[KanjiCardInput]) -> dict:
    """Create multiple kanji/vocab cards at once."""
    results = []
    errors = []
    for i, card in enumerate(cards):
        try:
            result = manager.create_kanji_card(
                kanji=card.kanji, reading=card.reading,
                meaning=card.meaning, tags=card.tags,
            )
            results.append({"note_id": result.note_id, "front": result.front})
        except Exception as e:
            errors.append({"index": i, "kanji": card.kanji, "error": str(e)})
    return {"created": len(results), "errors": errors, "cards": results}


@mcp.tool()
def create_manga_cards_batch(cards: list[MangaCardInput]) -> dict:
    """Create multiple manga vocab cards at once."""
    results = []
    errors = []
    for i, card in enumerate(cards):
        try:
            result = manager.create_manga_card(
                word=card.word, translation=card.translation,
                image_data=card.image_base64, image_url=card.image_url,
                tags=card.tags,
            )
            results.append({"note_id": result.note_id, "front": result.front})
        except Exception as e:
            errors.append({"index": i, "word": card.word, "error": str(e)})
    return {"created": len(results), "errors": errors, "cards": results}


@mcp.tool()
def sync_to_server() -> dict:
    """Sync the Anki collection and media to the self-hosted sync server.

    Call this after creating cards to make them available on other devices.
    """
    return sync.sync()


@mcp.tool()
def list_decks() -> list[dict]:
    """List all decks in the collection with note counts."""
    return manager.list_decks()


@mcp.tool()
def search_notes(query: str) -> list[dict]:
    """Search for existing notes using Anki search syntax.

    Examples: "deck:Japones KANJI", "tag:manga", or free text like "eat".
    Returns up to 50 matching notes.
    """
    return manager.search_notes(query)


@mcp.tool()
def get_collection_stats() -> dict:
    """Get an overview of the Anki collection (total notes, cards, decks, study stats)."""
    return manager.get_stats()


class SlashStripMiddleware:
    """Strip trailing slashes from request paths to avoid 307 redirects."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].endswith("/") and scope["path"] != "/":
            scope["path"] = scope["path"].rstrip("/")
        await self.app(scope, receive, send)


def create_app():
    """Create the Starlette ASGI app with secret-path auth.

    The MCP endpoint is mounted at /<token>/mcp/ — knowing the URL is the auth.
    This works with Claude Web (no custom headers) and Claude Code alike.
    """
    from starlette.applications import Starlette
    from starlette.middleware.cors import CORSMiddleware
    from starlette.routing import Mount

    mcp_app = mcp.http_app()
    token = settings.mcp_auth_token
    app = Starlette(
        routes=[Mount(f"/{token}", app=mcp_app)],
        lifespan=mcp_app.lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    # Strip trailing slashes to avoid FastMCP's internal 307 redirects
    # (which break behind TLS-terminating proxies like Tailscale Funnel)
    app.add_middleware(SlashStripMiddleware)
    return app


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
