from __future__ import annotations

import asyncio
import html
import logging
import secrets
import time
from dataclasses import dataclass, field

from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from .agent import CardAgent, PendingCard
from .anki_manager import AnkiManager
from .config import settings
from .sync_manager import SyncManager
from .word_extractor import CandidateExtraction, WordCandidate

logger = logging.getLogger(__name__)

manager = AnkiManager()
sync_mgr = SyncManager(manager)
agent = CardAgent()

# Lazy panel detector — only initialised when first image arrives
_panel_detector = None


def _get_panel_detector():
    global _panel_detector
    if _panel_detector is None:
        from pathlib import Path

        onnx_path = Path(settings.panel_model_path)
        if onnx_path.exists():
            from .panel_detector import OnnxPanelDetector

            _panel_detector = OnnxPanelDetector(model_path=str(onnx_path))
        else:
            from .panel_detector import PanelDetector

            _panel_detector = PanelDetector(device=settings.panel_model_device)
    return _panel_detector

bot = Bot(token=settings.telegram_bot_token)
dp = Dispatcher()

# Serialize access to Anki collection (not thread-safe)
agent_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Review session infrastructure
# ---------------------------------------------------------------------------

SESSION_TTL = 3600  # 1 hour


@dataclass
class ReviewSession:
    """Tracks a set of proposed cards awaiting user review."""
    cards: list[PendingCard]
    # status per card: None=pending, True=accepted, False=deleted
    status: list[bool | None] = field(default_factory=list)
    msg_ids: list[int] = field(default_factory=list)
    chat_id: int = 0
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.status:
            self.status = [None] * len(self.cards)

    @property
    def all_reviewed(self) -> bool:
        return all(s is not None for s in self.status)

    @property
    def pending_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.status) if s is None]


# In-memory store keyed by 8-char session ID
pending_reviews: dict[str, ReviewSession] = {}


@dataclass
class WordSelectionSession:
    """Tracks a candidate word list awaiting user selection (before translation)."""
    extraction: CandidateExtraction
    selected: list[bool] = field(default_factory=list)
    panel_msg_ids: dict[int, int] = field(default_factory=dict)  # panel_idx -> message_id
    control_msg_id: int = 0
    chat_id: int = 0
    caption: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.selected:
            self.selected = [False] * len(self.extraction.candidates)

    @property
    def n_selected(self) -> int:
        return sum(self.selected)


pending_word_selections: dict[str, WordSelectionSession] = {}


@dataclass
class ModeSession:
    """Holds an uploaded image while the user picks a workflow (read page / explain page)."""
    image_bytes: bytes
    caption: str
    chat_id: int
    created_at: float = field(default_factory=time.time)


pending_modes: dict[str, ModeSession] = {}


def _new_session_id() -> str:
    return secrets.token_hex(4)  # 8 hex chars


def _purge_stale_sessions() -> None:
    now = time.time()
    stale = [sid for sid, s in pending_reviews.items() if now - s.created_at > SESSION_TTL]
    for sid in stale:
        del pending_reviews[sid]
    stale_ws = [
        sid for sid, s in pending_word_selections.items()
        if now - s.created_at > SESSION_TTL
    ]
    for sid in stale_ws:
        del pending_word_selections[sid]
    stale_modes = [
        sid for sid, s in pending_modes.items()
        if now - s.created_at > SESSION_TTL
    ]
    for sid in stale_modes:
        del pending_modes[sid]


def _card_caption(card: PendingCard) -> str:
    if card.card_type == "kanji":
        return (
            f"<b>Front:</b> {card.kanji}\n"
            f"<b>Back:</b> {card.reading} ({card.meaning})"
        )
    reading_line = f"\n<b>Reading:</b> {card.reading}" if card.reading else ""
    return (
        f"<b>Word:</b> {card.word}{reading_line}\n"
        f"<b>Sentence:</b> {card.sentence}\n"
        f"<b>Translation:</b> {card.translation}"
    )


def _card_keyboard(session_id: str, index: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Accept", callback_data=f"mc:{session_id}:{index}:a"),
        InlineKeyboardButton(text="❌ Delete", callback_data=f"mc:{session_id}:{index}:d"),
    ]])


def _bulk_keyboard(session_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Accept All", callback_data=f"mc:{session_id}:all:a"),
            InlineKeyboardButton(text="❌ Delete All", callback_data=f"mc:{session_id}:all:d"),
        ],
        [
            InlineKeyboardButton(text="✅ Done — create accepted, skip rest", callback_data=f"mc:{session_id}:all:done"),
        ],
    ])


# ---------------------------------------------------------------------------
# Mode picker (shown right after photo upload)
# ---------------------------------------------------------------------------

def _mode_picker_keyboard(session_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔊 Read page", callback_data=f"mode:{session_id}:rp")],
        [InlineKeyboardButton(text="🧠 Explain page", callback_data=f"mode:{session_id}:ep")],
    ])


# ---------------------------------------------------------------------------
# Word-selection UI (step 1 of 2: pick which words to translate)
# ---------------------------------------------------------------------------

def _word_panel_keyboard(
    session_id: str, session: WordSelectionSession, panel_index: int,
) -> InlineKeyboardMarkup:
    """Per-candidate row: toggle button (left) + Explain button (right)."""
    rows = []
    for i, cand in enumerate(session.extraction.candidates):
        if cand.panel_index != panel_index:
            continue
        check = "☑" if session.selected[i] else "☐"
        label = f"{check} {cand.word} ({cand.reading})"
        if len(label) > 60:
            label = label[:59].rstrip() + "…"
        rows.append([
            InlineKeyboardButton(
                text=label,
                callback_data=f"ws:{session_id}:t:{i}",
            ),
            InlineKeyboardButton(
                text="💬 Explain",
                callback_data=f"ws:{session_id}:ex:{i}",
            ),
        ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _word_control_keyboard(session_id: str, n_selected: int) -> InlineKeyboardMarkup:
    label = f"✨ Generate {n_selected} cards" if n_selected > 0 else "✨ Generate (none selected)"
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="☑ Select all", callback_data=f"ws:{session_id}:all:s"),
            InlineKeyboardButton(text="☐ Clear all", callback_data=f"ws:{session_id}:all:c"),
        ],
        [InlineKeyboardButton(text=label, callback_data=f"ws:{session_id}:go:-")],
    ])


async def _send_word_picker(
    chat_id: int, session_id: str, session: WordSelectionSession,
) -> None:
    """One message per panel (image + word toggles), then a control message."""
    panel_indices = sorted({c.panel_index for c in session.extraction.candidates})
    for panel_idx in panel_indices:
        panel_cands = [c for c in session.extraction.candidates if c.panel_index == panel_idx]
        panel_image = panel_cands[0].panel_image
        ocr_lines = (
            session.extraction.ocr_per_panel[panel_idx]
            if panel_idx < len(session.extraction.ocr_per_panel) else []
        )
        ocr_text = "\n".join(f"<i>{line}</i>" for line in ocr_lines) if ocr_lines else "<i>(no text)</i>"
        caption = f"<b>Panel {panel_idx + 1}</b>\n{ocr_text}"
        if len(caption) > 1000:
            caption = caption[:997] + "..."
        kb = _word_panel_keyboard(session_id, session, panel_idx)
        photo = BufferedInputFile(panel_image, filename=f"panel_{panel_idx}.jpg")
        msg = await bot.send_photo(
            chat_id, photo=photo, caption=caption,
            parse_mode="HTML", reply_markup=kb,
        )
        session.panel_msg_ids[panel_idx] = msg.message_id

    n = len(session.extraction.candidates)
    control = await bot.send_message(
        chat_id,
        f"<b>{n}</b> candidate words across <b>{len(panel_indices)}</b> panels. "
        "Pick the ones to learn, then hit Generate.",
        parse_mode="HTML",
        reply_markup=_word_control_keyboard(session_id, 0),
    )
    session.control_msg_id = control.message_id


async def _send_card_previews(
    chat_id: int, session_id: str, session: ReviewSession
) -> None:
    """Send each proposed card as a photo+caption with inline keyboards."""
    for i, card in enumerate(session.cards):
        kb = _card_keyboard(session_id, i)
        caption = _card_caption(card)
        if card.image_data:
            photo = BufferedInputFile(card.image_data, filename=f"card_{i}.webp")
            msg = await bot.send_photo(
                chat_id, photo=photo, caption=caption,
                parse_mode="HTML", reply_markup=kb,
            )
        else:
            msg = await bot.send_message(
                chat_id, text=caption,
                parse_mode="HTML", reply_markup=kb,
            )
        session.msg_ids.append(msg.message_id)

    # Bulk buttons if more than one card
    if len(session.cards) > 1:
        bulk_msg = await bot.send_message(
            chat_id, text=f"{len(session.cards)} cards proposed — review above or use bulk actions:",
            reply_markup=_bulk_keyboard(session_id),
        )
        session.msg_ids.append(bulk_msg.message_id)


def _strip_html(text: str) -> str:
    """Remove HTML tags from text for TTS input."""
    import re
    return re.sub(r"<[^>]+>", "", text)


def _create_card(card: PendingCard) -> None:
    """Create the actual Anki card from a pending card."""
    if card.card_type == "kanji":
        manager.create_kanji_card(
            kanji=card.kanji, reading=card.reading,
            meaning=card.meaning, tags=card.tags,
        )
    else:
        # Generate TTS audio. Prefer the LLM-styled (emoji-decorated) Japanese
        # text, falling back to the plain sentence if styling is empty.
        audio_data = None
        tts_input = card.tts_text.strip() or _strip_html(card.sentence)
        if tts_input:
            try:
                from .tts import generate_tts
                audio_data = generate_tts(
                    tts_input,
                    caption=card.voice_description_jp.strip() or None,
                )
            except Exception as e:
                logger.warning("TTS generation failed for '%s': %s", card.word, e)
        manager.create_manga_card(
            word=card.word, sentence=card.sentence,
            translation=card.translation,
            image_data=card.image_data, reading=card.reading,
            audio_data=audio_data, tags=card.tags,
        )


async def _finalize_session(session_id: str, session: ReviewSession) -> None:
    """Pre-sync, create accepted cards in Anki, post-sync, send summary, clean up."""
    accepted_indices = [i for i, s in enumerate(session.status) if s is True]
    deleted = sum(1 for s in session.status if s is False)

    if accepted_indices:
        # Pre-sync: pull latest state so the local collection is up to date.
        # This prevents FULL_UPLOAD from overwriting the server with a stale
        # local collection that is missing the user's existing cards.
        pre_result = sync_mgr.pull()
        logger.info("Pre-sync before card creation: %s", pre_result["collection_sync"])

        for i in accepted_indices:
            _create_card(session.cards[i])

        sync_result = sync_mgr.push()
        sync_info = f"\nSync: {sync_result['collection_sync']}"
    else:
        sync_info = ""

    await bot.send_message(
        session.chat_id,
        f"Review complete: {len(accepted_indices)} accepted, {deleted} deleted.{sync_info}",
    )

    del pending_reviews[session_id]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _is_allowed(user_id: int, username: str | None = None) -> bool:
    ids = settings.allowed_user_ids
    names = settings.allowed_usernames
    if not ids and not names:
        return True  # no restrictions configured
    if ids and user_id in ids:
        return True
    if names and username and username.lower() in names:
        return True
    return False


@dp.message(F.text == "/start")
async def cmd_start(message: Message) -> None:
    if not _is_allowed(message.from_user.id, message.from_user.username):
        return
    await message.answer(
        "Welcome to Anki Bot!\n\n"
        "Send me text to create flashcards, or send a manga screenshot "
        "and I'll extract vocabulary from it.\n\n"
        "Commands:\n"
        "/stats - Collection statistics\n"
        "/decks - List all decks"
    )


@dp.message(F.text == "/stats")
async def cmd_stats(message: Message) -> None:
    if not _is_allowed(message.from_user.id, message.from_user.username):
        return
    stats = manager.get_stats()
    await message.answer(
        f"Collection Stats\n"
        f"Notes: {stats['total_notes']}\n"
        f"Cards: {stats['total_cards']}\n"
        f"Decks: {stats['total_decks']}\n"
        f"Studied today: {stats['studied_today']}"
    )


@dp.message(F.text == "/decks")
async def cmd_decks(message: Message) -> None:
    if not _is_allowed(message.from_user.id, message.from_user.username):
        return
    decks = manager.list_decks()
    if not decks:
        await message.answer("No decks found.")
        return
    lines = ["Decks:"]
    for d in decks:
        lines.append(f"  {d['name']}: {d['note_count']} notes")
    await message.answer("\n".join(lines))


@dp.message(F.photo)
async def handle_photo(message: Message) -> None:
    if not _is_allowed(message.from_user.id, message.from_user.username):
        return

    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    bio = await bot.download_file(file.file_path)
    image_bytes = bio.read()
    caption = message.caption or ""

    _purge_stale_sessions()
    session_id = _new_session_id()
    pending_modes[session_id] = ModeSession(
        image_bytes=image_bytes,
        caption=caption,
        chat_id=message.chat.id,
    )
    await message.answer(
        "What would you like to do with this image?",
        reply_markup=_mode_picker_keyboard(session_id),
    )


async def _run_read_page_flow(chat_id: int, image_bytes: bytes, caption: str) -> None:
    """Read Page pipeline: detect panels + OCR, style each panel, send image + TTS audio."""
    processing = await bot.send_message(chat_id, "Detecting panels and OCRing dialogue...")
    try:
        async with agent_lock:
            extraction = await agent.extract_candidates(image_bytes)
    except Exception as e:
        logger.exception("Read page extraction failed")
        await processing.delete()
        await bot.send_message(chat_id, f"Extraction failed: {e}")
        return

    if not extraction.panel_images:
        await processing.delete()
        await bot.send_message(chat_id, "No panels detected on this page.")
        return

    try:
        await processing.edit_text(
            f"Styling {len(extraction.panel_images)} panels for reading..."
        )
    except Exception:
        pass

    try:
        async with agent_lock:
            styled = await agent.style_panels_for_reading(extraction)
    except Exception as e:
        logger.exception("Read page styling failed")
        await processing.delete()
        await bot.send_message(chat_id, f"Styling failed: {e}")
        return

    await processing.delete()

    n_panels = len(extraction.panel_images)

    async def _tts_for_panel(i: int) -> bytes | None:
        item = styled[i] if i < len(styled) else None
        ocr_lines = (
            extraction.ocr_per_panel[i]
            if i < len(extraction.ocr_per_panel) else []
        )
        tts_input = (item.tts_text.strip() if item else "") or " ".join(ocr_lines).strip()
        if not tts_input:
            return None
        try:
            from .tts import generate_tts
            return await asyncio.to_thread(
                generate_tts,
                tts_input,
                caption=(item.voice_description_jp.strip() if item else None) or None,
            )
        except Exception as e:
            logger.warning("TTS generation failed for panel %d: %s", i + 1, e)
            return None

    # Fan out all TTS jobs in parallel; we await each one in panel order below
    # so audio messages stay aligned with their photos.
    tts_tasks = [asyncio.create_task(_tts_for_panel(i)) for i in range(n_panels)]

    for i, panel_image in enumerate(extraction.panel_images):
        ocr_lines = (
            extraction.ocr_per_panel[i]
            if i < len(extraction.ocr_per_panel) else []
        )
        ocr_text = (
            "\n".join(f"<i>{line}</i>" for line in ocr_lines)
            if ocr_lines else "<i>(no dialogue)</i>"
        )
        panel_caption = f"<b>Panel {i + 1}/{n_panels}</b>\n{ocr_text}"
        if len(panel_caption) > 1000:
            panel_caption = panel_caption[:997] + "..."
        photo = BufferedInputFile(panel_image, filename=f"panel_{i}.webp")
        await bot.send_photo(chat_id, photo=photo, caption=panel_caption, parse_mode="HTML")

        audio_bytes = await tts_tasks[i]
        if audio_bytes is None:
            continue
        audio_file = BufferedInputFile(audio_bytes, filename=f"panel_{i}.wav")
        await bot.send_audio(
            chat_id,
            audio=audio_file,
            title=f"Panel {i + 1}",
            performer="Read Page",
        )

    await bot.send_message(chat_id, f"Read Page complete: {n_panels} panels.")


async def _run_explain_page_flow(chat_id: int, image_bytes: bytes, caption: str) -> None:
    """Explain Page pipeline: detect panels + OCR, ask the LLM for a summary
    + vocab + expressions, then surface vocab/expressions through the
    existing word-picker UI so the user can build flashcards from them."""
    processing = await bot.send_message(chat_id, "Detecting panels and OCRing dialogue...")
    try:
        async with agent_lock:
            extraction = await agent.extract_candidates(image_bytes)
    except Exception as e:
        logger.exception("Explain page extraction failed")
        await processing.delete()
        await bot.send_message(chat_id, f"Extraction failed: {e}")
        return

    if not any(extraction.ocr_per_panel):
        await processing.delete()
        await bot.send_message(chat_id, "No text detected on this page.")
        return

    try:
        await processing.edit_text("Asking the LLM to explain the page...")
    except Exception:
        pass

    try:
        async with agent_lock:
            explanation = await agent.explain_page(extraction)
    except Exception as e:
        logger.exception("Explain page LLM call failed")
        await processing.delete()
        await bot.send_message(chat_id, f"Explanation failed: {e}")
        return

    await processing.delete()
    await _send_explanation(chat_id, explanation)

    synthetic = _explanation_to_extraction(explanation, extraction)
    if not synthetic.candidates:
        await bot.send_message(chat_id, "No flashcard-worthy vocabulary on this page.")
        return

    _purge_stale_sessions()
    session_id = _new_session_id()
    session = WordSelectionSession(
        extraction=synthetic, chat_id=chat_id, caption=caption,
    )
    pending_word_selections[session_id] = session
    await _send_word_picker(chat_id, session_id, session)


TELEGRAM_MSG_LIMIT = 4096


async def _send_explanation(chat_id: int, explanation) -> None:
    """Render the full PageExplanation (summary + vocab + expressions) into a
    single self-contained message — splitting by section only if Telegram's
    4096-char limit would be exceeded."""
    summary_block = _format_summary_block(explanation.summary)
    vocab_block = _format_vocab_block(explanation.vocabulary or [])
    expr_block = _format_expressions_block(explanation.expressions or [])

    blocks = [b for b in (summary_block, vocab_block, expr_block) if b]
    if not blocks:
        return

    combined = "\n\n".join(blocks)
    if len(combined) <= TELEGRAM_MSG_LIMIT:
        await bot.send_message(chat_id, combined, parse_mode="HTML")
        return

    for block in blocks:
        await bot.send_message(chat_id, block, parse_mode="HTML")


def _format_summary_block(summary: str) -> str:
    summary = (summary or "").strip()
    if not summary:
        return ""
    return f"<b>📖 Summary</b>\n{html.escape(summary)}"


def _format_vocab_block(vocabulary: list) -> str:
    lines: list[str] = []
    for v in vocabulary:
        word = html.escape((v.word or "").strip())
        reading = html.escape((v.reading or "").strip())
        gloss = html.escape((v.translation or "").strip())
        note = html.escape((v.note or "").strip())
        if not word:
            continue
        header = f"• <b>{word}</b>"
        if reading and reading != word:
            header += f" ({reading})"
        if gloss:
            header += f" — {gloss}"
        lines.append(header)
        if note:
            lines.append(f"   <i>{note}</i>")
    if not lines:
        return ""
    return "<b>📚 Vocabulary</b>\n" + "\n".join(lines)


def _format_expressions_block(expressions: list) -> str:
    lines: list[str] = []
    for e in expressions:
        expr = html.escape((e.expression or "").strip())
        reading = html.escape((e.reading or "").strip())
        explanation_text = html.escape((e.explanation or "").strip())
        if not expr:
            continue
        header = f"• <b>{expr}</b>"
        if reading and reading != expr:
            header += f" ({reading})"
        lines.append(header)
        if explanation_text:
            lines.append(f"   {explanation_text}")
    if not lines:
        return ""
    return "<b>💡 Expressions</b>\n" + "\n".join(lines)


# Kanji range for the okurigana-stripping fallback in _match_panel_for_surface.
_KANJI_RE = "一-鿿"


def _match_panel_for_surface(
    surface: str, ocr_per_panel: list[list[str]]
) -> tuple[int, str]:
    """Find the best (panel_index, sentence) for a surface form.

    1. Exact substring match on any OCR line.
    2. If the surface has kanji, retry on its kanji-only prefix (handles cases
       where the LLM returned the bare stem and the page has ~okurigana~).
    3. Fallback: panel 0 with sentence == surface so the validator in
       agent.generate_cards (`surface in sentence`) trivially passes.
    """
    for p_idx, lines in enumerate(ocr_per_panel):
        for line in lines:
            if surface and surface in line:
                return p_idx, line
    import re
    kanji_only = "".join(re.findall(f"[{_KANJI_RE}]+", surface))
    if kanji_only and kanji_only != surface:
        for p_idx, lines in enumerate(ocr_per_panel):
            for line in lines:
                if kanji_only in line:
                    return p_idx, line
    return 0, surface


def _explanation_to_extraction(
    explanation, extraction: CandidateExtraction,
) -> CandidateExtraction:
    """Adapt PageExplanation.{vocabulary, expressions} into a synthetic
    CandidateExtraction so they flow through the existing word-picker /
    flashcard pipeline. Reuses the original page's OCR + panel images."""
    triples: list[tuple[str, str, str]] = []  # (surface, reading, translation)
    for v in explanation.vocabulary or []:
        triples.append((v.word.strip(), v.reading.strip(), v.translation.strip()))
    for e in explanation.expressions or []:
        triples.append((e.expression.strip(), e.reading.strip(), e.explanation.strip()))

    seen: set[str] = set()
    candidates: list[WordCandidate] = []
    for surface, reading, translation in triples:
        if not surface or surface in seen:
            continue
        seen.add(surface)
        panel_idx, sentence = _match_panel_for_surface(surface, extraction.ocr_per_panel)
        if panel_idx >= len(extraction.panel_images):
            panel_idx = 0
        panel_image = (
            extraction.panel_images[panel_idx]
            if extraction.panel_images else b""
        )
        candidates.append(WordCandidate(
            word=surface,
            surface=surface,
            reading=reading,
            sentence=sentence,
            panel_index=panel_idx,
            panel_image=panel_image,
            translation=translation,
        ))

    return CandidateExtraction(
        candidates=candidates,
        ocr_per_panel=extraction.ocr_per_panel,
        panel_images=extraction.panel_images,
    )


@dp.message(F.text)
async def handle_text(message: Message) -> None:
    if not _is_allowed(message.from_user.id, message.from_user.username):
        return
    # Skip unknown commands
    if message.text.startswith("/"):
        return

    processing = await message.answer("Thinking...")
    async with agent_lock:
        try:
            result = await agent.process_text(message.text)
        except Exception as e:
            logger.exception("Agent error")
            await processing.delete()
            await message.answer(f"Error: {e}")
            return
    await processing.delete()

    if result.text:
        await message.answer(result.text)

    # If there are proposed cards, start a review session
    if result.pending_cards:
        _purge_stale_sessions()
        session_id = _new_session_id()
        session = ReviewSession(
            cards=result.pending_cards,
            chat_id=message.chat.id,
        )
        pending_reviews[session_id] = session
        await _send_card_previews(message.chat.id, session_id, session)


@dp.callback_query(F.data.startswith("mode:"))
async def handle_mode_pick(callback: CallbackQuery) -> None:
    """User chose flashcards vs. read-page right after uploading an image."""
    parts = callback.data.split(":")
    if len(parts) != 3:
        await callback.answer("Invalid callback data.")
        return

    _, session_id, choice = parts
    mode_session = pending_modes.pop(session_id, None)
    if mode_session is None:
        await callback.answer("Session expired.", show_alert=True)
        return

    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

    if choice == "rp":
        try:
            await callback.message.edit_text("🔊 Reading page...")
        except Exception:
            pass
        await callback.answer()
        await _run_read_page_flow(
            mode_session.chat_id, mode_session.image_bytes, mode_session.caption,
        )
        return

    if choice == "ep":
        try:
            await callback.message.edit_text("🧠 Explaining page...")
        except Exception:
            pass
        await callback.answer()
        await _run_explain_page_flow(
            mode_session.chat_id, mode_session.image_bytes, mode_session.caption,
        )
        return

    await callback.answer("Unknown choice.")


@dp.callback_query(F.data.startswith("mc:"))
async def handle_card_review(callback: CallbackQuery) -> None:
    """Handle Accept/Delete button presses for card review."""
    parts = callback.data.split(":")
    if len(parts) != 4:
        await callback.answer("Invalid callback data.")
        return

    _, session_id, index_str, action = parts

    session = pending_reviews.get(session_id)
    if session is None:
        await callback.answer("Session expired.", show_alert=True)
        return

    # --- Bulk action ---
    if index_str == "all":
        remaining = session.pending_indices
        if not remaining:
            await callback.answer("All cards already reviewed.")
            return

        if action == "done":
            # "Done" — skip all remaining cards, finalize with what was accepted
            async with agent_lock:
                for i in remaining:
                    session.status[i] = False
                    card = session.cards[i]
                    caption = _card_caption(card) + "\n\n⏭ Skipped"
                    msg_id = session.msg_ids[i]
                    try:
                        if card.image_data:
                            await bot.edit_message_caption(
                                chat_id=session.chat_id, message_id=msg_id,
                                caption=caption, parse_mode="HTML",
                            )
                        else:
                            await bot.edit_message_text(
                                chat_id=session.chat_id, message_id=msg_id,
                                text=caption, parse_mode="HTML",
                            )
                    except Exception:
                        pass

            try:
                await callback.message.edit_reply_markup(reply_markup=None)
            except Exception:
                pass

            accepted = sum(1 for s in session.status if s is True)
            await callback.answer(f"Skipped {len(remaining)}, creating {accepted} accepted cards.")
            await _finalize_session(session_id, session)
            return

        accept = action == "a"

        async with agent_lock:
            for i in remaining:
                session.status[i] = accept
                # Update individual card message
                status_text = "✅ Accepted" if accept else "❌ Deleted"
                card = session.cards[i]
                caption = _card_caption(card) + f"\n\n{status_text}"
                msg_id = session.msg_ids[i]
                try:
                    if card.image_data:
                        await bot.edit_message_caption(
                            chat_id=session.chat_id, message_id=msg_id,
                            caption=caption, parse_mode="HTML",
                        )
                    else:
                        await bot.edit_message_text(
                            chat_id=session.chat_id, message_id=msg_id,
                            text=caption, parse_mode="HTML",
                        )
                except Exception:
                    pass  # message may already be edited

        # Remove bulk keyboard
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass

        action_word = "Accepted" if accept else "Deleted"
        await callback.answer(f"{action_word} {len(remaining)} cards.")

        if session.all_reviewed:
            await _finalize_session(session_id, session)
        return

    # --- Single card action ---
    accept = action == "a"
    index = int(index_str)
    if index < 0 or index >= len(session.cards):
        await callback.answer("Invalid card index.")
        return

    if session.status[index] is not None:
        await callback.answer("Already reviewed.")
        return

    card = session.cards[index]

    async with agent_lock:
        session.status[index] = accept

    # Update the message to show result and remove keyboard
    status_text = "✅ Accepted" if accept else "❌ Deleted"
    caption = _card_caption(card) + f"\n\n{status_text}"
    try:
        if card.image_data:
            await callback.message.edit_caption(
                caption=caption, parse_mode="HTML",
            )
        else:
            await callback.message.edit_text(
                text=caption, parse_mode="HTML",
            )
    except Exception:
        pass

    await callback.answer(status_text)

    # Check if all cards reviewed
    if session.all_reviewed:
        # Remove bulk keyboard message if it exists
        if len(session.cards) > 1 and len(session.msg_ids) > len(session.cards):
            try:
                bulk_msg_id = session.msg_ids[-1]
                await bot.edit_message_text(
                    chat_id=session.chat_id, message_id=bulk_msg_id,
                    text="All cards reviewed.",
                )
            except Exception:
                pass
        await _finalize_session(session_id, session)


@dp.callback_query(F.data.startswith("ws:"))
async def handle_word_selection(callback: CallbackQuery) -> None:
    """Toggle/bulk/go buttons on the word-selection picker."""
    parts = callback.data.split(":")
    if len(parts) < 3:
        await callback.answer("Invalid callback data.")
        return

    session_id = parts[1]
    action = parts[2]
    session = pending_word_selections.get(session_id)
    if session is None:
        await callback.answer("Session expired.", show_alert=True)
        return

    if action == "t":
        if len(parts) != 4:
            await callback.answer("Invalid callback data.")
            return
        idx = int(parts[3])
        if idx < 0 or idx >= len(session.extraction.candidates):
            await callback.answer("Invalid index.")
            return
        session.selected[idx] = not session.selected[idx]
        panel_idx = session.extraction.candidates[idx].panel_index
        msg_id = session.panel_msg_ids.get(panel_idx)
        if msg_id is not None:
            try:
                await bot.edit_message_reply_markup(
                    chat_id=session.chat_id, message_id=msg_id,
                    reply_markup=_word_panel_keyboard(session_id, session, panel_idx),
                )
            except Exception:
                pass
        try:
            await bot.edit_message_reply_markup(
                chat_id=session.chat_id, message_id=session.control_msg_id,
                reply_markup=_word_control_keyboard(session_id, session.n_selected),
            )
        except Exception:
            pass
        await callback.answer()
        return

    if action == "all":
        if len(parts) != 4:
            await callback.answer("Invalid callback data.")
            return
        new_state = parts[3] == "s"
        session.selected = [new_state] * len(session.selected)
        for panel_idx, msg_id in session.panel_msg_ids.items():
            try:
                await bot.edit_message_reply_markup(
                    chat_id=session.chat_id, message_id=msg_id,
                    reply_markup=_word_panel_keyboard(session_id, session, panel_idx),
                )
            except Exception:
                pass
        try:
            await bot.edit_message_reply_markup(
                chat_id=session.chat_id, message_id=session.control_msg_id,
                reply_markup=_word_control_keyboard(session_id, session.n_selected),
            )
        except Exception:
            pass
        await callback.answer("Selected all." if new_state else "Cleared all.")
        return

    if action == "go":
        if session.n_selected == 0:
            await callback.answer("Pick at least one word first.", show_alert=True)
            return

        n = session.n_selected
        try:
            await bot.edit_message_text(
                chat_id=session.chat_id, message_id=session.control_msg_id,
                text=f"Generating cards for {n} selected word{'s' if n != 1 else ''}...",
            )
        except Exception:
            pass
        await callback.answer()

        selected_indices = [i for i, s in enumerate(session.selected) if s]
        try:
            async with agent_lock:
                cards = await agent.generate_cards(session.extraction, selected_indices)
        except Exception as e:
            logger.exception("Card generation failed")
            await bot.send_message(session.chat_id, f"Card generation failed: {e}")
            return

        del pending_word_selections[session_id]

        if not cards:
            await bot.send_message(
                session.chat_id,
                "All selected candidates failed validation. No cards to review.",
            )
            return

        review_session_id = _new_session_id()
        review_session = ReviewSession(cards=cards, chat_id=session.chat_id)
        pending_reviews[review_session_id] = review_session
        await _send_card_previews(session.chat_id, review_session_id, review_session)
        return

    if action == "ex":
        if len(parts) != 4:
            await callback.answer("Invalid callback data.")
            return
        idx = int(parts[3])
        if idx < 0 or idx >= len(session.extraction.candidates):
            await callback.answer("Invalid index.")
            return
        await callback.answer()
        await _run_word_explain(session, idx)
        return

    await callback.answer("Unknown action.")


async def _run_word_explain(session: WordSelectionSession, idx: int) -> None:
    """Generate a simple-Japanese explanation of one word + narrator TTS, then
    send it back as a voice note. Does not mutate the picker session."""
    cand = session.extraction.candidates[idx]
    status = await bot.send_message(
        session.chat_id, f"🤔 Explaining 「{cand.word}」…",
    )
    try:
        async with agent_lock:
            explanation = await agent.explain_word(
                cand, session.extraction.ocr_per_panel,
            )
    except Exception as e:
        logger.exception("explain_word LLM call failed")
        try:
            await status.edit_text(f"Explain failed for 「{cand.word}」: {e}")
        except Exception:
            pass
        return

    tts_text = (explanation.tts_text or "").strip()
    if not tts_text:
        try:
            await status.edit_text(f"Explain returned empty text for 「{cand.word}」.")
        except Exception:
            pass
        return

    try:
        from .tts import generate_tts
        audio_data = await asyncio.to_thread(
            generate_tts,
            tts_text,
            caption=(explanation.voice_description_jp or "").strip() or None,
        )
    except Exception as e:
        logger.exception("explain_word TTS failed")
        try:
            await status.edit_text(f"TTS failed for 「{cand.word}」: {e}")
        except Exception:
            pass
        return

    try:
        await status.delete()
    except Exception:
        pass

    voice_file = BufferedInputFile(audio_data, filename=f"explain_{idx}.ogg")
    caption = f"💬 <b>{html.escape(cand.word)}</b> ({html.escape(cand.reading)})"
    try:
        await bot.send_voice(
            session.chat_id, voice=voice_file, caption=caption, parse_mode="HTML",
        )
    except Exception:
        audio_fallback = BufferedInputFile(audio_data, filename=f"explain_{idx}.wav")
        await bot.send_audio(
            session.chat_id, audio=audio_fallback,
            title=f"Explain {cand.word}", performer="Explain",
            caption=caption, parse_mode="HTML",
        )


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Anki Telegram Bot...")
    try:
        await dp.start_polling(bot)
    finally:
        manager.close()


if __name__ == "__main__":
    asyncio.run(main())
