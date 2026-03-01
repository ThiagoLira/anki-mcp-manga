from __future__ import annotations

import asyncio
import io
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

from .agent import PendingCard, build_agent
from .anki_manager import AnkiManager
from .config import settings
from .sync_manager import SyncManager

logger = logging.getLogger(__name__)

manager = AnkiManager()
sync_mgr = SyncManager(manager)
run_agent = build_agent(manager)

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


def _new_session_id() -> str:
    return secrets.token_hex(4)  # 8 hex chars


def _purge_stale_sessions() -> None:
    now = time.time()
    stale = [sid for sid, s in pending_reviews.items() if now - s.created_at > SESSION_TTL]
    for sid in stale:
        del pending_reviews[sid]


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
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Accept All", callback_data=f"mc:{session_id}:all:a"),
        InlineKeyboardButton(text="❌ Delete All", callback_data=f"mc:{session_id}:all:d"),
    ]])


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
        # Generate TTS audio from the Japanese sentence
        audio_data = None
        if card.sentence:
            try:
                from .tts import generate_tts
                plain_sentence = _strip_html(card.sentence)
                audio_data = generate_tts(plain_sentence)
            except Exception as e:
                logger.warning("TTS generation failed for '%s': %s", card.word, e)
        manager.create_manga_card(
            word=card.word, sentence=card.sentence,
            translation=card.translation,
            image_data=card.image_data, reading=card.reading,
            audio_data=audio_data, tags=card.tags,
        )


async def _finalize_session(session_id: str, session: ReviewSession) -> None:
    """Create accepted cards in Anki, sync, send summary, clean up."""
    accepted = sum(1 for s in session.status if s is True)
    deleted = sum(1 for s in session.status if s is False)

    if accepted > 0:
        sync_result = sync_mgr.sync()
        sync_info = f"\nSync: {sync_result['collection_sync']}"
    else:
        sync_info = ""

    await bot.send_message(
        session.chat_id,
        f"Review complete: {accepted} accepted, {deleted} deleted.{sync_info}",
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

    # Download the largest resolution photo
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    bio = await bot.download_file(file.file_path)
    image_bytes = bio.read()

    caption = message.caption or "Extract vocabulary from this manga page and create cards."

    # Panel detection (optional, runs before the agent)
    page_analysis = None
    if settings.enable_panel_detection:
        processing = await message.answer("Detecting panels...")
        try:
            detector = _get_panel_detector()
            page_analysis = detector.detect(image_bytes)
            logger.info("Detected %d panels", len(page_analysis.panels))
        except Exception as e:
            logger.warning("Panel detection failed, falling back to full page: %s", e)
            page_analysis = None
        await processing.delete()

    processing = await message.answer("Processing image...")
    async with agent_lock:
        try:
            result = await run_agent(caption, image_bytes, page_analysis)
        except Exception as e:
            logger.exception("Agent error")
            await processing.delete()
            await message.answer(f"Error: {e}")
            return
    await processing.delete()

    # Send agent text response
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
            result = await run_agent(message.text)
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

    accept = action == "a"

    # --- Bulk action ---
    if index_str == "all":
        remaining = session.pending_indices
        if not remaining:
            await callback.answer("All cards already reviewed.")
            return

        async with agent_lock:
            for i in remaining:
                session.status[i] = accept
                if accept:
                    _create_card(session.cards[i])
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
        if accept:
            _create_card(card)

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


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Anki Telegram Bot...")
    try:
        await dp.start_polling(bot)
    finally:
        manager.close()


if __name__ == "__main__":
    asyncio.run(main())
