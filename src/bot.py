from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message

from .agent import build_agent
from .anki_manager import AnkiManager
from .config import settings
from .sync_manager import SyncManager

logger = logging.getLogger(__name__)

manager = AnkiManager()
sync_mgr = SyncManager(manager)
run_agent = build_agent(manager, sync_mgr)

bot = Bot(token=settings.telegram_bot_token)
dp = Dispatcher()

# Serialize access to Anki collection (not thread-safe)
agent_lock = asyncio.Lock()


def _is_allowed(user_id: int) -> bool:
    allowed = settings.allowed_user_ids
    return not allowed or user_id in allowed


@dp.message(F.text == "/start")
async def cmd_start(message: Message) -> None:
    if not _is_allowed(message.from_user.id):
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
    if not _is_allowed(message.from_user.id):
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
    if not _is_allowed(message.from_user.id):
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
    if not _is_allowed(message.from_user.id):
        return

    # Download the largest resolution photo
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    bio = await bot.download_file(file.file_path)
    image_bytes = bio.read()

    caption = message.caption or "Extract vocabulary from this manga page and create cards."

    processing = await message.answer("Processing image...")
    async with agent_lock:
        try:
            response = await run_agent(caption, image_bytes)
        except Exception as e:
            logger.exception("Agent error")
            response = f"Error: {e}"
    await processing.delete()
    await message.answer(response)


@dp.message(F.text)
async def handle_text(message: Message) -> None:
    if not _is_allowed(message.from_user.id):
        return
    # Skip unknown commands
    if message.text.startswith("/"):
        return

    processing = await message.answer("Thinking...")
    async with agent_lock:
        try:
            response = await run_agent(message.text)
        except Exception as e:
            logger.exception("Agent error")
            response = f"Error: {e}"
    await processing.delete()
    await message.answer(response)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Anki Telegram Bot...")
    try:
        await dp.start_polling(bot)
    finally:
        manager.close()


if __name__ == "__main__":
    asyncio.run(main())
