from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .config import settings

if TYPE_CHECKING:
    from .panel_detector import PageAnalysis

logger = logging.getLogger(__name__)

MULTI_PANEL_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Data structures (kept for bot.py review flow)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

class MangaProposal(BaseModel):
    """A proposed manga vocabulary flashcard."""
    word: str = Field(description="The bare vocabulary word (e.g. 規則)")
    reading: str = Field(description="Hiragana reading of the word (e.g. きそく)")
    sentence: str = Field(
        description="Full Japanese sentence with target word in <b> tags "
        '(e.g. "<b>規則</b>を守れ")'
    )
    translation: str = Field(
        description="Full sentence translation with translated word in <b> tags "
        '(e.g. "Follow the <b>rules</b>")'
    )
    panel_number: int | None = Field(
        None, description="0-based panel index matching the ①②③ labels"
    )


class MangaExtraction(BaseModel):
    """Vocabulary extracted from a manga page or panel."""
    summary: str = Field(description="Brief transcription and narrative summary")
    cards: list[MangaProposal] = Field(description="Proposed vocabulary cards")


class KanjiProposal(BaseModel):
    """A proposed kanji/vocab flashcard."""
    kanji: str = Field(description="The kanji or vocabulary word")
    reading: str = Field(description="Hiragana reading")
    meaning: str = Field(description="English meaning")


class TextResponse(BaseModel):
    """Response to a text-only request."""
    response: str = Field(description="Text response to the user")
    kanji_cards: list[KanjiProposal] = Field(
        default_factory=list, description="Proposed kanji cards, if requested"
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

MANGA_PROMPT = """\
You are a Japanese language study assistant that creates Anki flashcards \
from manga pages.

When panels are detected, the image has panels numbered ①②③... in manga \
reading order (right-to-left, top-to-bottom).

Follow this two-pass process:
1. **Transcribe first**: Read ALL dialogue, referencing panel numbers \
①②③... to establish full context and reading order.
2. **Create cards**: Extract interesting vocabulary at or above JLPT N4 level. \
Skip basic N5/N4 words (e.g. 食べる, 大きい, 学校) — the user already knows those.

For each card, provide:
- `word`: the bare vocabulary word
- `reading`: hiragana reading (e.g. きそく for 規則)
- `sentence`: full Japanese sentence with target word in <b>bold</b>
- `translation`: full sentence translation with translated word in <b>bold</b>
- `panel_number`: 0-based index matching the ①②③ labels (when panels are visible)

Write a brief `summary` covering transcription and narrative context.
Respond in English."""

SUMMARY_PROMPT = """\
You are a Japanese language expert. The image shows a manga page with panels \
numbered ①②③... in reading order (right-to-left, top-to-bottom).

Provide a concise summary:
1. Transcribe ALL dialogue per panel.
2. Summarize the narrative: who speaks, what happens, emotional context.

This will be used as context for per-panel vocabulary extraction."""

PER_PANEL_PROMPT = """\
You are a Japanese language study assistant that creates Anki flashcards.

Extract interesting vocabulary from this single manga panel. \
Skip basic JLPT N5/N4 words — the user already knows those. Focus on N3+ vocabulary.

For each card, provide:
- `word`: the bare vocabulary word
- `reading`: hiragana reading
- `sentence`: full Japanese sentence with target word in <b>bold</b>
- `translation`: full sentence translation with translated word in <b>bold</b>

Do NOT set panel_number — the image is already the correct panel.
Use the page context below to understand implied subjects or references.

## Page context
{summary}"""

TEXT_PROMPT = """\
You are a Japanese language study assistant that creates Anki flashcards.
If the user asks to create kanji/vocab cards, include them in kanji_cards.
Otherwise, just respond helpfully about Japanese language topics.
Respond in English."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_data_uri(image_bytes: bytes) -> str:
    """Build a base64 data URI with correct MIME type."""
    b64 = base64.b64encode(image_bytes).decode()
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        mime = "image/webp"
    elif image_bytes[:2] == b"\xff\xd8":
        mime = "image/jpeg"
    elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    else:
        mime = "image/webp"
    return f"data:{mime};base64,{b64}"


def _image_content(image_bytes: bytes, text: str) -> list[dict[str, Any]]:
    """Build a multimodal message content list (image + text)."""
    return [
        {"type": "image_url", "image_url": {"url": _image_data_uri(image_bytes)}},
        {"type": "text", "text": text},
    ]


# ---------------------------------------------------------------------------
# CardAgent
# ---------------------------------------------------------------------------

class CardAgent:
    """Orchestrates LLM calls for manga vocabulary extraction and card creation."""

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=settings.openrouter_model,
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )

    async def process_image(
        self,
        caption: str,
        image_bytes: bytes,
        page_analysis: PageAnalysis | None = None,
    ) -> AgentResult:
        """Process a manga image and extract vocabulary cards."""
        n_panels = len(page_analysis.panels) if page_analysis else 0
        logger.info(
            "process_image: caption=%d chars, image=%d bytes, panels=%d",
            len(caption), len(image_bytes), n_panels,
        )

        if page_analysis and n_panels >= MULTI_PANEL_THRESHOLD:
            logger.info("Using multi-panel path (>= %d panels)", MULTI_PANEL_THRESHOLD)
            return await self._multi_panel(caption, image_bytes, page_analysis)

        logger.info("Using single-pass path")
        return await self._single_pass(caption, image_bytes, page_analysis)

    async def process_text(self, text: str) -> AgentResult:
        """Handle a text-only message (kanji cards, questions)."""
        logger.info("process_text: %s", text[:100])
        llm = self._llm.with_structured_output(TextResponse)
        result = await llm.ainvoke([
            SystemMessage(content=TEXT_PROMPT),
            HumanMessage(content=text),
        ])
        logger.info(
            "process_text: response=%d chars, kanji_cards=%d",
            len(result.response), len(result.kanji_cards),
        )
        cards = [
            PendingCard(
                card_type="kanji",
                kanji=c.kanji, reading=c.reading, meaning=c.meaning,
            )
            for c in result.kanji_cards
        ]
        return AgentResult(text=result.response, pending_cards=cards)

    # --- Private: processing paths ---

    async def _single_pass(
        self,
        caption: str,
        image_bytes: bytes,
        page_analysis: PageAnalysis | None,
    ) -> AgentResult:
        """Single LLM call for a full page (<= threshold panels)."""
        if page_analysis:
            llm_image = page_analysis.annotated_image
            panels = {i: p.image_bytes for i, p in enumerate(page_analysis.panels)}
        else:
            llm_image = image_bytes
            panels = {}

        extraction = await self._extract_manga(llm_image, caption, MANGA_PROMPT)
        cards = self._build_manga_cards(extraction, panels, fallback_image=image_bytes)
        logger.info("single_pass complete: %d cards", len(cards))
        return AgentResult(text=extraction.summary, pending_cards=cards)

    async def _multi_panel(
        self,
        caption: str,
        image_bytes: bytes,
        page_analysis: PageAnalysis,
    ) -> AgentResult:
        """Summary + per-panel extraction for pages with many panels."""
        # Step 1: summarise the full page
        summary = await self._summarize_page(page_analysis.annotated_image, caption)
        logger.info("Page summary:\n%s", summary)

        # Step 2: extract per panel, tracking seen words to avoid duplicates
        all_cards: list[PendingCard] = []
        panel_texts: list[str] = []
        seen_words: set[str] = set()
        base_prompt = PER_PANEL_PROMPT.format(summary=summary)

        for i, panel in enumerate(page_analysis.panels):
            logger.info("Extracting panel %d/%d", i + 1, len(page_analysis.panels))
            if seen_words:
                skip_line = "\n\nAlready extracted (do NOT repeat): " + ", ".join(sorted(seen_words))
                prompt = base_prompt + skip_line
            else:
                prompt = base_prompt
            extraction = await self._extract_manga(
                panel.image_bytes,
                "Extract vocabulary from this panel and propose cards.",
                prompt,
            )
            cards = self._build_manga_cards(extraction, {}, fallback_image=panel.image_bytes)
            all_cards.extend(cards)
            panel_texts.append(extraction.summary)
            seen_words.update(c.word for c in cards)

        text = (
            f"Processed {len(page_analysis.panels)} panels.\n\n"
            + "\n\n".join(panel_texts)
        )
        logger.info("multi_panel complete: %d cards total", len(all_cards))
        return AgentResult(text=text, pending_cards=all_cards)

    # --- Private: LLM calls ---

    async def _extract_manga(
        self, image: bytes, user_text: str, system_prompt: str,
    ) -> MangaExtraction:
        """Send image + text to LLM and get structured manga extraction."""
        llm = self._llm.with_structured_output(MangaExtraction)
        result = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=_image_content(image, user_text)),
        ])
        words = [c.word for c in result.cards]
        logger.info("extract_manga: %d cards [%s]", len(result.cards), ", ".join(words))
        return result

    async def _summarize_page(self, annotated_image: bytes, caption: str) -> str:
        """Get a narrative summary of the full annotated page."""
        result = await self._llm.ainvoke([
            SystemMessage(content=SUMMARY_PROMPT),
            HumanMessage(content=_image_content(annotated_image, caption)),
        ])
        return result.content

    # --- Private: card building ---

    @staticmethod
    def _build_manga_cards(
        extraction: MangaExtraction,
        panels: dict[int, bytes],
        fallback_image: bytes,
    ) -> list[PendingCard]:
        """Convert structured extraction into PendingCards with correct images."""
        cards: list[PendingCard] = []
        for proposal in extraction.cards:
            pn = proposal.panel_number
            if panels and pn is not None and pn in panels:
                image = panels[pn]
                logger.info("  card '%s': using panel %d image", proposal.word, pn)
            else:
                image = fallback_image
                logger.info("  card '%s': using fallback image", proposal.word)
            cards.append(PendingCard(
                card_type="manga",
                word=proposal.word,
                reading=proposal.reading,
                sentence=proposal.sentence,
                translation=proposal.translation,
                image_data=image,
            ))
        return cards
