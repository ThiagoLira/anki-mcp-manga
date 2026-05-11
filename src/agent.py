from __future__ import annotations

import asyncio
import base64
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .config import settings

if TYPE_CHECKING:
    from .word_extractor import CandidateExtraction, WordCandidate

logger = logging.getLogger(__name__)


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
    # TTS styling (populated for manga cards by the styled-translation pass)
    tts_text: str = ""               # Japanese sentence with inline style emojis
    voice_description_jp: str = ""   # JP caption for VoiceDesign-mode TTS


@dataclass
class AgentResult:
    """Result from running the agent."""
    text: str
    pending_cards: list[PendingCard] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

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


class StyledTranslationItem(BaseModel):
    """Translation + TTS styling for a single sentence."""
    translation: str = Field(
        description="Natural English translation of the sentence with the target word in <b>...</b>."
    )
    tts_text: str = Field(
        description=(
            "The original Japanese sentence (no HTML tags) with style emojis "
            "added inline. Used as TTS input."
        )
    )
    voice_description_jp: str = Field(
        description=(
            "Short Japanese caption describing the speaker's voice "
            "(gender/age/tone). Used as the VoiceDesign caption."
        )
    )


class StyledTranslations(BaseModel):
    """Batched styled-translation response — same length and order as input."""
    items: list[StyledTranslationItem]


class StyledPanelItem(BaseModel):
    """TTS styling for a full panel (one combined utterance, one voice)."""
    tts_text: str = Field(
        description=(
            "The panel's dialogue/narration concatenated into a single Japanese "
            "block (no HTML tags) with style emojis added inline. Used as TTS "
            "input. Empty string if the panel has no dialogue."
        )
    )
    voice_description_jp: str = Field(
        description=(
            "Short Japanese caption describing the panel speaker's voice "
            "(gender/age/tone). Used as the VoiceDesign caption. Empty string "
            "if the panel has no dialogue."
        )
    )


class StyledPanels(BaseModel):
    """Batched styled-panel response — same length and order as input."""
    items: list[StyledPanelItem]


class VocabItem(BaseModel):
    """One vocabulary entry for the Explain Page workflow."""
    word: str = Field(
        description="The word as it appears on the page (kanji or kana)."
    )
    reading: str = Field(
        description="Hiragana reading. Always provide, even for kana-only words."
    )
    translation: str = Field(
        description="Short English gloss — a few words to one short clause."
    )
    note: str = Field(
        default="",
        description=(
            "Optional context: nuance, why a learner might miss this, the "
            "panel/scene it appears in. Empty if no extra context is needed."
        ),
    )


class ExpressionItem(BaseModel):
    """One non-obvious expression (idiom, slang, set phrase) for Explain Page."""
    expression: str = Field(
        description="The expression as it appears on the page."
    )
    reading: str = Field(
        description="Hiragana reading of the whole expression."
    )
    explanation: str = Field(
        description=(
            "What the expression means and why it's non-obvious "
            "(idiomatic, slang, regional, set phrase, etc.) in English."
        )
    )


class PageExplanation(BaseModel):
    """Page-level explanation for the Explain Page workflow."""
    summary: str = Field(
        description=(
            "2-4 sentences in English describing what is happening on the page: "
            "characters, setting, conflict, emotional beat. No spoilers beyond "
            "what is visible."
        )
    )
    vocabulary: list[VocabItem] = Field(
        description=(
            "5-15 difficult words at intermediate (N3/N2) level. Skip N5/N4 "
            "basics. Skip names of people and places."
        )
    )
    expressions: list[ExpressionItem] = Field(
        default_factory=list,
        description=(
            "0-5 non-obvious expressions (idioms, slang, set phrases). "
            "Empty list is fine if nothing notable."
        )
    )


class WordExplanation(BaseModel):
    """Per-word in-scene explanation for the Explain button on the picker."""
    tts_text: str = Field(
        description=(
            "2-4 sentences in simple Japanese (around JLPT N4) explaining the "
            "word's meaning and tying it to what is happening on the current "
            "manga panel. Used as TTS input."
        )
    )
    voice_description_jp: str = Field(
        description=(
            "Short Japanese caption for a clear, neutral narrator voice. "
            "Used as the VoiceDesign caption."
        )
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TEXT_PROMPT = """\
You are a Japanese language study assistant that creates Anki flashcards.
If the user asks to create kanji/vocab cards, include them in kanji_cards.
Otherwise, just respond helpfully about Japanese language topics.
Respond in English."""

TRANSLATION_PROMPT = """\
Process Japanese sentences from a manga page. For each sentence return three \
fields: an English translation, an emoji-styled Japanese version for TTS, and \
a Japanese voice description for the speaker.

## PAGE TRANSCRIPT (context — speakers, tone, narrative)
{transcript}

## SENTENCES TO PROCESS
{numbered_pairs}

For each numbered sentence, return:

1. translation
   Natural English translation. Wrap the English equivalent of the marked \
word in <b>...</b>.

2. tts_text
   The original Japanese sentence (no HTML tags) with style emojis added \
inline to convey emotion and prosody. Place emojis adjacent to the words/\
phrases they modify. Most lines need 0–2 emojis; repeat an emoji to intensify.
   ONLY use emojis from this set:
   - Emotion: 😠 angry, 😏 teasing, 🥺 timid, 🫶 gentle, 😱 scream, 😖 in pain, \
😟 worried, 🫣 shy, 🙄 exasperated, 😊 cheerful, 🙏 pleading, 🥴 drunk, \
😰 panicked/stuttering, 😌 relieved, 🤔 questioning, 😲 surprise, 😆 joyful, \
🤭 chuckle, 😭 sobbing
   - Breath/voice: 😮‍💨 sigh, 🌬️ heavy breath, 😮 gasp, 😪 sleepy, 🥱 yawn, \
🥵 panting, 👂 whisper, 📢 loud/echo, 👅 wet sound, 💋 lip smack, 🥤 swallow, \
🤧 sniffle, 😒 tongue click, 🤐 muffled
   - Prosody: ⏩ fast, 🐢 slow, ⏸️ pause, 📞 phone/speaker
   - Sound: 👌 backchannel, 🎵 humming
   Do NOT invent new emojis or use any not in this list.

3. voice_description_jp
   A short Japanese caption describing the speaker's voice (gender, age, \
tone). Use the page transcript to infer who the speaker is. End with \
「読み上げてください。」 or 「語ってください。」. Keep under 80 characters.
   Examples:
   - 「30代の落ち着いた成人男性の声で、自信に満ちた穏やかな口調で読み上げてください。」
   - 「年配の威厳ある男性の声で、深く重く力強くゆっくりと語ってください。」
   - 「元気な少年の高めの声で、無邪気で好奇心を込めて読み上げてください。」
   - 「神秘的な大人の女性の声で、低めに落ち着いた口調で読み上げてください。」

Output JSON {{"items": [{{"translation": "...", "tts_text": "...", \
"voice_description_jp": "..."}}, ...]}} — same length and order as the input."""

READ_PAGE_PROMPT = """\
Process Japanese dialogue from a manga page panel-by-panel for a TTS reading \
exercise. For each panel return styled TTS text and a voice description.

## PAGE TRANSCRIPT (context — speakers, tone, narrative)
{transcript}

For each numbered panel below, return:

1. tts_text
   The panel's dialogue concatenated into a single Japanese utterance (no HTML \
tags). If the panel has multiple bubbles, join them with full-width spaces \
「　」 or punctuation that fits the flow. Add style emojis inline to convey \
emotion and prosody. Place emojis adjacent to the words/phrases they modify. \
Most panels need 0–3 emojis; repeat an emoji to intensify.
   ONLY use emojis from this set:
   - Emotion: 😠 angry, 😏 teasing, 🥺 timid, 🫶 gentle, 😱 scream, 😖 in pain, \
😟 worried, 🫣 shy, 🙄 exasperated, 😊 cheerful, 🙏 pleading, 🥴 drunk, \
😰 panicked/stuttering, 😌 relieved, 🤔 questioning, 😲 surprise, 😆 joyful, \
🤭 chuckle, 😭 sobbing
   - Breath/voice: 😮‍💨 sigh, 🌬️ heavy breath, 😮 gasp, 😪 sleepy, 🥱 yawn, \
🥵 panting, 👂 whisper, 📢 loud/echo, 👅 wet sound, 💋 lip smack, 🥤 swallow, \
🤧 sniffle, 😒 tongue click, 🤐 muffled
   - Prosody: ⏩ fast, 🐢 slow, ⏸️ pause, 📞 phone/speaker
   - Sound: 👌 backchannel, 🎵 humming
   Do NOT invent new emojis or use any not in this list.
   If the panel has no dialogue, return an empty string.

2. voice_description_jp
   A short Japanese caption describing the panel speaker's voice (gender, \
age, tone). Use the page transcript to infer who is speaking. End with \
「読み上げてください。」 or 「語ってください。」. Keep under 80 characters. \
If the panel has multiple speakers, pick the dominant one. \
If the panel has no dialogue, return an empty string.
   Examples:
   - 「30代の落ち着いた成人男性の声で、自信に満ちた穏やかな口調で読み上げてください。」
   - 「年配の威厳ある男性の声で、深く重く力強くゆっくりと語ってください。」
   - 「元気な少年の高めの声で、無邪気で好奇心を込めて読み上げてください。」
   - 「神秘的な大人の女性の声で、低めに落ち着いた口調で読み上げてください。」

## PANELS
{numbered_panels}

Output JSON {{"items": [{{"tts_text": "...", "voice_description_jp": "..."}}, \
...]}} — same length and order as the input panels."""

EXPLAIN_PAGE_PROMPT = """\
You are helping an intermediate Japanese learner (around JLPT N3/N2 level) \
understand a manga page they just read. Below is the OCR'd dialogue, panel by \
panel, in reading order.

## PAGE TRANSCRIPT
{transcript}

Return three things:

1. summary
   2-4 sentences in English describing what's happening on this page: who is \
speaking, the setting/situation, the emotional beat, any conflict or shift. \
Do not invent details that aren't supported by the dialogue. If the page is \
ambiguous, say so briefly. No spoilers beyond what's on the page.

2. vocabulary
   A list of 5-15 difficult words from the dialogue. Calibrate for an N3/N2 \
learner — skip N5/N4 basics (e.g. 食べる, 大きい, 学校, 言う), skip proper \
nouns (character/place names), skip onomatopoeia unless it's a non-obvious \
mimetic. Prefer words that are useful to learn (common in adult media) over \
extreme rarities. For each item:
   - word: as written on the page (kanji or kana)
   - reading: hiragana reading (always provide; for kana-only words, repeat the kana)
   - translation: short English gloss
   - note: optional one-line context (nuance, why a learner might miss it, what \
panel it shows up in). Leave empty if not useful.

3. expressions
   A list of 0-5 less-obvious expressions: idioms, slang, regional speech, \
set phrases, indirect speech where the literal reading misleads. Skip if \
nothing on the page qualifies. For each:
   - expression: as written
   - reading: hiragana reading of the whole expression
   - explanation: what it actually means and why a learner who knows the \
individual words might still miss it.

Write summary, translation, note, and explanation in plain English (no \
markdown, no HTML tags). Keep individual entries short — these are quick \
reference notes, not essays."""

EXPLAIN_WORD_PROMPT = """\
You are a Japanese narrator helping an intermediate learner understand one \
word from a manga page they just read. The audio is for listening practice, \
so the explanation must be in clear, simple Japanese — never English.

## TARGET WORD
{word} ({reading})

## CURRENT PANEL DIALOGUE
{panel_dialogue}

## FULL PAGE TRANSCRIPT (context)
{transcript}

Return two fields:

1. tts_text
   2-4 short sentences in plain Japanese, calibrated to JLPT N4 vocabulary \
and grammar. Do NOT use English. Avoid katakana loanwords when defining the \
word (use native Japanese paraphrases instead). Structure:
   a. A simple definition of {word} in Japanese.
   b. A tiny in-context paraphrase or example.
   c. One sentence tying the word to what is happening in the current panel \
on this manga page.
   Use minimal emojis — at most one or two from the set ⏸️ (pause), 🐢 (slow), \
⏩ (fast), 📢 (loud), 👂 (whisper). Do NOT use emotion emojis: this is a \
narrator, not a character. If unsure, use none.
   No HTML tags. No markdown.

2. voice_description_jp
   A short Japanese caption for a clear, neutral narrator voice (no \
character voicing, no acting). Under 80 characters, ending with \
「読み上げてください。」 or 「解説してください。」.
   Examples:
   - 「明瞭で落ち着いた中性的なナレーターの声で、はっきりと優しく解説してください。」
   - 「丁寧でクリアな解説者の声で、ゆっくりと分かりやすく読み上げてください。」"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe_local_llm(url: str, timeout: float) -> bool:
    """Quick HEAD/GET to {url}/models to see if a local OpenAI-compatible server
    is alive. Any network/HTTP failure returns False (caller falls back)."""
    import urllib.error
    import urllib.request

    probe_url = f"{url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(probe_url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


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

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        kwargs = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # If the caller didn't pin model/base_url/api_key, probe the local LLM
        # (llama-server on janus over the tailnet) and prefer it when reachable.
        # Falls back to OpenRouter on probe failure.
        if model is None and base_url is None and api_key is None:
            if _probe_local_llm(settings.local_llm_url, settings.local_llm_probe_timeout_s):
                model = settings.local_llm_model
                base_url = settings.local_llm_url
                api_key = "local"
                logger.info("Using local LLM at %s", base_url)
            else:
                logger.info(
                    "Local LLM at %s unreachable; using OpenRouter",
                    settings.local_llm_url,
                )

        self._llm = ChatOpenAI(
            model=model or settings.openrouter_model,
            base_url=base_url or "https://openrouter.ai/api/v1",
            api_key=api_key or settings.openrouter_api_key,
            **kwargs,
        )
        self._word_extractor = None  # lazy

    # ------------------------------------------------------------------
    # Two-phase pipeline (used by the bot's word-selection flow)
    # ------------------------------------------------------------------

    async def extract_candidates(self, image_bytes: bytes) -> "CandidateExtraction":
        """Run the deterministic word-extraction pipeline. No LLM calls.

        Heavy CPU work (panel detection, bubble detection, manga-ocr,
        tokenization) is offloaded to a worker thread so the event loop
        stays responsive.
        """
        if self._word_extractor is None:
            from .word_extractor import WordExtractor
            self._word_extractor = WordExtractor()
        logger.info("extract_candidates: image=%d bytes", len(image_bytes))
        return await asyncio.to_thread(self._word_extractor.extract, image_bytes)

    async def generate_cards(
        self,
        extraction: "CandidateExtraction",
        selected_indices: list[int],
    ) -> list[PendingCard]:
        """Translate the selected candidates (one batched LLM call) and build PendingCards."""
        selected = [extraction.candidates[i] for i in selected_indices]
        if not selected:
            return []

        transcript = self._format_transcript(extraction.ocr_per_panel)
        pairs = "\n".join(
            f"{i + 1}. word=「{c.word}」 (form in sentence: 「{c.surface}」), "
            f"sentence=「{c.sentence}」"
            for i, c in enumerate(selected)
        )
        prompt = TRANSLATION_PROMPT.format(transcript=transcript, numbered_pairs=pairs)

        logger.info(
            "generate_cards: translating+styling %d candidates in one batched call",
            len(selected),
        )
        llm = self._llm.with_structured_output(StyledTranslations)
        result = await llm.ainvoke([HumanMessage(content=prompt)])

        items = result.items
        if len(items) != len(selected):
            logger.warning(
                "Item count mismatch: got %d, expected %d — padding/truncating",
                len(items), len(selected),
            )
            empty = StyledTranslationItem(
                translation="", tts_text="", voice_description_jp="",
            )
            items = (list(items) + [empty] * len(selected))[:len(selected)]

        cards: list[PendingCard] = []
        for cand, item in zip(selected, items):
            sentence_bolded = self._bold(cand.sentence, cand.surface)
            sent_clean = re.sub(r"</?b>", "", sentence_bolded)
            if (unicodedata.normalize("NFKC", cand.surface)
                    not in unicodedata.normalize("NFKC", sent_clean)):
                logger.warning("Dropping %s: surface %s not in sentence", cand.word, cand.surface)
                continue
            translation = (item.translation or "").strip()
            if not translation or not any(c.isascii() and c.isalpha() for c in translation):
                logger.warning("Dropping %s: empty/non-English translation", cand.word)
                continue
            tts_text = (item.tts_text or "").strip() or sent_clean
            voice_description = (item.voice_description_jp or "").strip()
            cards.append(PendingCard(
                card_type="manga",
                word=cand.word,
                reading=cand.reading,
                sentence=sentence_bolded,
                translation=translation,
                image_data=cand.panel_image,
                tts_text=tts_text,
                voice_description_jp=voice_description,
            ))
        logger.info("generate_cards: produced %d cards (dropped %d)",
                    len(cards), len(selected) - len(cards))
        return cards

    async def style_panels_for_reading(
        self,
        extraction: "CandidateExtraction",
    ) -> list[StyledPanelItem]:
        """Batched LLM call: style each panel's dialogue for TTS reading.

        Returns one StyledPanelItem per panel in `extraction.ocr_per_panel`,
        in the same order. Panels with no dialogue get empty fields.
        """
        n_panels = len(extraction.ocr_per_panel)
        if n_panels == 0:
            return []

        transcript = self._format_transcript(extraction.ocr_per_panel)
        numbered = []
        for i, lines in enumerate(extraction.ocr_per_panel):
            joined = " / ".join(lines) if lines else "(no dialogue)"
            numbered.append(f"{i + 1}. {joined}")
        prompt = READ_PAGE_PROMPT.format(
            transcript=transcript,
            numbered_panels="\n".join(numbered),
        )

        logger.info("style_panels_for_reading: styling %d panels in one batched call", n_panels)
        llm = self._llm.with_structured_output(StyledPanels)
        result = await llm.ainvoke([HumanMessage(content=prompt)])

        items = list(result.items)
        if len(items) != n_panels:
            logger.warning(
                "Panel count mismatch: got %d, expected %d — padding/truncating",
                len(items), n_panels,
            )
            empty = StyledPanelItem(tts_text="", voice_description_jp="")
            items = (items + [empty] * n_panels)[:n_panels]
        return items

    async def explain_page(
        self,
        extraction: "CandidateExtraction",
    ) -> PageExplanation:
        """Single-shot LLM call: page summary + vocab list + expressions.

        Targeted at intermediate (N3/N2) Japanese learners. Returns a
        PageExplanation; raises if the LLM call fails.
        """
        transcript = self._format_transcript(extraction.ocr_per_panel)
        prompt = EXPLAIN_PAGE_PROMPT.format(transcript=transcript)
        logger.info(
            "explain_page: explaining %d panels in one call",
            len(extraction.ocr_per_panel),
        )
        llm = self._llm.with_structured_output(PageExplanation)
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        logger.info(
            "explain_page: %d vocab, %d expressions",
            len(result.vocabulary), len(result.expressions),
        )
        return result

    async def explain_word(
        self,
        candidate: "WordCandidate",
        ocr_per_panel: list[list[str]],
    ) -> WordExplanation:
        """Single LLM call: simple-Japanese in-scene explanation of one word."""
        transcript = self._format_transcript(ocr_per_panel)
        panel_dialogue = "(no dialogue)"
        if 0 <= candidate.panel_index < len(ocr_per_panel):
            panel_lines = ocr_per_panel[candidate.panel_index]
            if panel_lines:
                panel_dialogue = " / ".join(panel_lines)
        prompt = EXPLAIN_WORD_PROMPT.format(
            word=candidate.word,
            reading=candidate.reading,
            panel_dialogue=panel_dialogue,
            transcript=transcript,
        )
        logger.info(
            "explain_word: word=%s reading=%s panel=%d",
            candidate.word, candidate.reading, candidate.panel_index,
        )
        llm = self._llm.with_structured_output(WordExplanation)
        return await llm.ainvoke([HumanMessage(content=prompt)])

    @staticmethod
    def _format_transcript(ocr_per_panel: list[list[str]]) -> str:
        parts = []
        for i, lines in enumerate(ocr_per_panel):
            joined = " / ".join(lines) if lines else "(no text)"
            parts.append(f"Panel {i + 1}: {joined}")
        return "\n".join(parts)

    @staticmethod
    def _bold(sentence: str, surface: str) -> str:
        return re.sub(re.escape(surface), f"<b>{surface}</b>", sentence, count=1)

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
