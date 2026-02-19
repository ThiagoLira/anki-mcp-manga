from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .config import settings

if TYPE_CHECKING:
    from .anki_manager import AnkiManager
    from .sync_manager import SyncManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a Japanese language study assistant that creates Anki flashcards.
You have tools to create two types of cards:

## Kanji cards (deck: Japones KANJI)
For learning kanji and vocabulary words:
- Front: the kanji or word
- Back: reading (hiragana) + meaning
Use create_kanji_card for individual cards, create_kanji_cards_batch for multiple.

## Manga vocab cards (deck: Japones Vocab Mangas)
For vocabulary extracted from manga pages. These are context-rich cards:
- Front: manga page screenshot + Japanese sentence with the target word in <b>bold</b>
- Back: full sentence translation with the translated target word in <b>bold</b>

Example: if the word is 規則 from the sentence 規則を守れ:
- sentence: "<b>規則</b>を守れ"
- translation: "Follow the <b>rules</b>"

Use create_manga_card for individual cards, create_manga_cards_batch for multiple.

## Guidelines
- When the user sends a manga screenshot, read the text in the image, pick out \
interesting vocabulary, and create manga vocab cards with the image attached.
- The `word` field is just the bare vocabulary word (for search/identification).
- The `sentence` field is the full Japanese sentence with the target word wrapped in <b> tags.
- The `translation` field is the full sentence translation with the target word wrapped in <b> tags.
- After creating cards, ALWAYS call sync_to_server to push changes.
- Respond in the same language as the user (Portuguese or English typically).
"""

RunAgent = Callable[[str, bytes | None], Coroutine[Any, Any, str]]


def build_agent(
    manager: AnkiManager, sync_mgr: SyncManager
) -> RunAgent:
    """Build the LangGraph ReAct agent with tools bound to the given managers."""

    # Mutable dict to hold current image bytes — tools read from here
    image_store: dict[str, bytes] = {}

    @tool
    def create_kanji_card(
        kanji: str, reading: str, meaning: str, tags: list[str] | None = None
    ) -> str:
        """Create a kanji/vocab flashcard. Front: kanji. Back: reading + meaning."""
        result = manager.create_kanji_card(
            kanji=kanji, reading=reading, meaning=meaning, tags=tags
        )
        return f"Created kanji card: {result.front} (note_id={result.note_id})"

    @tool
    def create_manga_card(
        word: str,
        sentence: str,
        translation: str,
        attach_image: bool = True,
        tags: list[str] | None = None,
    ) -> str:
        """Create a manga vocab flashcard.
        Front: manga image + Japanese sentence (target word in <b>bold</b>).
        Back: full sentence translation (target word in <b>bold</b>).
        Set attach_image=False to skip attaching the current image."""
        image_bytes = image_store.get("current") if attach_image else None
        result = manager.create_manga_card(
            word=word, sentence=sentence, translation=translation,
            image_data=image_bytes, tags=tags,
        )
        img_status = " with image" if image_bytes else ""
        return f"Created manga card: {result.front}{img_status} (note_id={result.note_id})"

    @tool
    def create_kanji_cards_batch(cards_json: str) -> str:
        """Create multiple kanji cards at once.
        cards_json is a JSON array of objects with keys: kanji, reading, meaning, tags (optional)."""
        cards = json.loads(cards_json)
        results = []
        errors = []
        for i, card in enumerate(cards):
            try:
                result = manager.create_kanji_card(
                    kanji=card["kanji"],
                    reading=card["reading"],
                    meaning=card["meaning"],
                    tags=card.get("tags"),
                )
                results.append(result.front)
            except Exception as e:
                errors.append(f"Card {i} ({card.get('kanji', '?')}): {e}")
        msg = f"Created {len(results)} kanji cards: {', '.join(results)}"
        if errors:
            msg += f"\nErrors: {'; '.join(errors)}"
        return msg

    @tool
    def create_manga_cards_batch(
        cards_json: str, attach_image: bool = True
    ) -> str:
        """Create multiple manga vocab cards at once.
        cards_json is a JSON array of objects with keys: word, sentence, translation, tags (optional).
        sentence should have the target word in <b>bold</b>.
        translation should have the translated word in <b>bold</b>.
        Set attach_image=False to skip attaching the current image."""
        cards = json.loads(cards_json)
        image_bytes = image_store.get("current") if attach_image else None
        results = []
        errors = []
        for i, card in enumerate(cards):
            try:
                result = manager.create_manga_card(
                    word=card["word"],
                    sentence=card["sentence"],
                    translation=card["translation"],
                    image_data=image_bytes,
                    tags=card.get("tags"),
                )
                results.append(result.front)
            except Exception as e:
                errors.append(f"Card {i} ({card.get('word', '?')}): {e}")
        msg = f"Created {len(results)} manga cards: {', '.join(results)}"
        if errors:
            msg += f"\nErrors: {'; '.join(errors)}"
        return msg

    @tool
    def sync_to_server() -> str:
        """Sync the Anki collection and media to the self-hosted sync server.
        Call this after creating cards."""
        result = sync_mgr.sync()
        return f"Synced — collection: {result['collection_sync']}, media: {result['media_sync']}"

    @tool
    def search_notes(query: str) -> str:
        """Search for existing notes using Anki search syntax.
        Examples: 'deck:Japones KANJI', 'tag:manga', or free text like 'eat'."""
        results = manager.search_notes(query)
        if not results:
            return "No notes found."
        lines = []
        for r in results:
            fields = ", ".join(f"{k}={v}" for k, v in r["fields"].items() if v)
            lines.append(f"[{r['id']}] {fields} tags={r['tags']}")
        return "\n".join(lines)

    @tool
    def list_decks() -> str:
        """List all decks in the Anki collection with note counts."""
        decks = manager.list_decks()
        if not decks:
            return "No decks found."
        return "\n".join(f"- {d['name']}: {d['note_count']} notes" for d in decks)

    tools = [
        create_kanji_card,
        create_manga_card,
        create_kanji_cards_batch,
        create_manga_cards_batch,
        sync_to_server,
        search_notes,
        list_decks,
    ]

    llm = ChatOpenAI(
        model=settings.openrouter_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )

    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    async def run_agent(text: str, image_bytes: bytes | None = None) -> str:
        """Run the agent with a user message, optionally including an image."""
        if image_bytes:
            image_store["current"] = image_bytes
        else:
            image_store.pop("current", None)

        # Build the message content
        content: list[dict[str, Any]] = []
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        if text:
            content.append({"type": "text", "text": text})

        message = HumanMessage(content=content if len(content) > 1 else text or "")

        result = await agent.ainvoke({"messages": [message]})

        # Extract the last AI message
        ai_messages = [m for m in result["messages"] if m.type == "ai" and m.content]
        if ai_messages:
            return ai_messages[-1].content
        return "Done."

    return run_agent
