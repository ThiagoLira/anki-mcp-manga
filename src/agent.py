from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .config import settings

if TYPE_CHECKING:
    from .anki_manager import AnkiManager
    from .panel_detector import PageAnalysis

logger = logging.getLogger(__name__)


@dataclass
class PendingCard:
    """A proposed card awaiting user review."""
    card_type: str  # "manga" or "kanji"
    # Manga fields
    word: str = ""
    sentence: str = ""
    translation: str = ""
    image_data: bytes | None = None
    tags: list[str] | None = None
    # Kanji fields
    kanji: str = ""
    reading: str = ""
    meaning: str = ""


@dataclass
class AgentResult:
    """Result from running the agent."""
    text: str
    pending_cards: list[PendingCard] = field(default_factory=list)

SYSTEM_PROMPT = """\
You are a Japanese language study assistant that creates Anki flashcards.
You have tools to propose two types of cards for user review:

## Kanji cards (deck: Japones KANJI)
For learning kanji and vocabulary words:
- Front: the kanji or word
- Back: reading (hiragana) + meaning
Use propose_kanji_card for individual cards, propose_kanji_cards_batch for multiple.
These tools PROPOSE cards for user review — they are NOT created in Anki yet.

## Manga vocab cards (deck: Japones Vocab Mangas)
For vocabulary extracted from manga pages. These are context-rich cards:
- Front: manga panel screenshot + Japanese sentence with the target word in <b>bold</b>
- Back: full sentence translation with the translated target word in <b>bold</b>

Example: if the word is 規則 from the sentence 規則を守れ:
- sentence: "<b>規則</b>を守れ"
- translation: "Follow the <b>rules</b>"

Use propose_manga_card for individual cards, propose_manga_cards_batch for multiple.
These tools PROPOSE cards for user review — they are NOT created in Anki yet.

## Panel-based workflow
When panels are detected, the image you see has panels numbered ①②③... in manga \
reading order (right-to-left, top-to-bottom). Each card will be attached with the \
cropped panel image instead of the full page.

Follow this two-pass process:
1. **Transcribe first**: Read ALL dialogue in the image, referencing panel numbers \
①②③... to establish full context and reading order.
2. **Create cards**: Extract interesting vocabulary and propose cards. For each card, \
specify `panel_number` (0-based index matching the ①②③ labels) so the card gets \
the correct cropped panel image.

## Guidelines
- When the user sends a manga screenshot, read the text in the image, pick out \
interesting vocabulary, and propose manga vocab cards with the image attached.
- The `word` field is just the bare vocabulary word (for search/identification).
- The `sentence` field is the full Japanese sentence with the target word wrapped in <b> tags.
- The `translation` field is the full sentence translation with the target word wrapped in <b> tags.
- Respond in English.
"""

RunAgent = Callable[[str, bytes | None, "PageAnalysis | None"], Coroutine[Any, Any, AgentResult]]


def build_agent(manager: AnkiManager) -> RunAgent:
    """Build the LangGraph ReAct agent with tools bound to the given managers."""

    # Mutable dict to hold current image bytes — tools read from here
    image_store: dict[str, Any] = {}
    # Accumulates proposed manga cards during a single run
    pending_cards: list[PendingCard] = []

    @tool
    def propose_kanji_card(
        kanji: str, reading: str, meaning: str, tags: list[str] | None = None
    ) -> str:
        """Propose a kanji/vocab flashcard for user review (not created in Anki yet).
        Front: kanji. Back: reading + meaning."""
        card = PendingCard(
            card_type="kanji", kanji=kanji, reading=reading,
            meaning=meaning, tags=tags,
        )
        pending_cards.append(card)
        return f"Proposed kanji card: {kanji}"

    @tool
    def propose_manga_card(
        word: str,
        sentence: str,
        translation: str,
        panel_number: int | None = None,
        attach_image: bool = True,
        tags: list[str] | None = None,
    ) -> str:
        """Propose a manga vocab flashcard for user review (not created in Anki yet).
        Front: manga image + Japanese sentence (target word in <b>bold</b>).
        Back: full sentence translation (target word in <b>bold</b>).
        Set panel_number (0-based) to attach the cropped panel image instead of the full page.
        Set attach_image=False to skip attaching any image."""
        image_bytes: bytes | None = None
        if attach_image:
            panels = image_store.get("panels")
            if panel_number is not None and panels and 0 <= panel_number < len(panels):
                image_bytes = panels[panel_number]
            else:
                image_bytes = image_store.get("current")
        card = PendingCard(
            card_type="manga", word=word, sentence=sentence,
            translation=translation, image_data=image_bytes, tags=tags,
        )
        pending_cards.append(card)
        panel_info = f" (panel {panel_number})" if panel_number is not None and image_bytes else ""
        img_status = f" with image{panel_info}" if image_bytes else ""
        return f"Proposed manga card: {word}{img_status}"

    @tool
    def propose_kanji_cards_batch(cards_json: str) -> str:
        """Propose multiple kanji cards at once for user review (not created in Anki yet).
        cards_json is a JSON array of objects with keys: kanji, reading, meaning, tags (optional)."""
        cards = json.loads(cards_json)
        proposed = []
        errors = []
        for i, card_data in enumerate(cards):
            try:
                card = PendingCard(
                    card_type="kanji",
                    kanji=card_data["kanji"],
                    reading=card_data["reading"],
                    meaning=card_data["meaning"],
                    tags=card_data.get("tags"),
                )
                pending_cards.append(card)
                proposed.append(card.kanji)
            except Exception as e:
                errors.append(f"Card {i} ({card_data.get('kanji', '?')}): {e}")
        msg = f"Proposed {len(proposed)} kanji cards: {', '.join(proposed)}"
        if errors:
            msg += f"\nErrors: {'; '.join(errors)}"
        return msg

    @tool
    def propose_manga_cards_batch(
        cards_json: str, attach_image: bool = True
    ) -> str:
        """Propose multiple manga vocab cards at once for user review (not created in Anki yet).
        cards_json is a JSON array of objects with keys:
          word, sentence, translation, panel_number (optional, 0-based), tags (optional).
        sentence should have the target word in <b>bold</b>.
        translation should have the translated word in <b>bold</b>.
        Set attach_image=False to skip attaching images."""
        cards = json.loads(cards_json)
        panels = image_store.get("panels")
        fallback_image = image_store.get("current")
        proposed = []
        errors = []
        for i, card in enumerate(cards):
            try:
                image_bytes: bytes | None = None
                if attach_image:
                    pn = card.get("panel_number")
                    if pn is not None and panels and 0 <= pn < len(panels):
                        image_bytes = panels[pn]
                    else:
                        image_bytes = fallback_image
                pending = PendingCard(
                    card_type="manga",
                    word=card["word"],
                    sentence=card["sentence"],
                    translation=card["translation"],
                    image_data=image_bytes,
                    tags=card.get("tags"),
                )
                pending_cards.append(pending)
                proposed.append(pending.word)
            except Exception as e:
                errors.append(f"Card {i} ({card.get('word', '?')}): {e}")
        msg = f"Proposed {len(proposed)} manga cards: {', '.join(proposed)}"
        if errors:
            msg += f"\nErrors: {'; '.join(errors)}"
        return msg

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
        propose_kanji_card,
        propose_manga_card,
        propose_kanji_cards_batch,
        propose_manga_cards_batch,
        search_notes,
        list_decks,
    ]

    llm = ChatOpenAI(
        model=settings.openrouter_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )

    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    async def run_agent(
        text: str,
        image_bytes: bytes | None = None,
        page_analysis: PageAnalysis | None = None,
    ) -> AgentResult:
        """Run the agent with a user message, optionally including an image.

        When page_analysis is provided, the annotated image (with panel numbers)
        is sent to the LLM, and cropped panels are stored for card attachment.
        """
        # Clear pending cards from any previous run
        pending_cards.clear()

        # Determine which image the LLM sees vs. what gets attached to cards
        if page_analysis:
            # LLM sees annotated image with ①②③ labels
            llm_image = page_analysis.annotated_image
            # Cards get clean cropped panels
            image_store["panels"] = [p.image_bytes for p in page_analysis.panels]
            image_store["current"] = image_bytes  # fallback: original image
        elif image_bytes:
            llm_image = image_bytes
            image_store["current"] = image_bytes
            image_store.pop("panels", None)
        else:
            llm_image = None
            image_store.pop("current", None)
            image_store.pop("panels", None)

        # Build the message content
        content: list[dict[str, Any]] = []
        if llm_image:
            b64 = base64.b64encode(llm_image).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{b64}"},
                }
            )
        if text:
            content.append({"type": "text", "text": text})

        message = HumanMessage(content=content if len(content) > 1 else text or "")

        result = await agent.ainvoke({"messages": [message]})

        # Extract the last AI message
        ai_messages = [m for m in result["messages"] if m.type == "ai" and m.content]
        text = ai_messages[-1].content if ai_messages else "Done."
        return AgentResult(text=text, pending_cards=list(pending_cards))

    return run_agent
