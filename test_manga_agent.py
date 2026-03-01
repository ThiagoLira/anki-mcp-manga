import asyncio
import base64
import io
import logging
from pathlib import Path

from PIL import Image

# Important: this loads .env and its credentials under the hood
from src.config import settings
from src.agent import AgentResult, PendingCard, build_agent
from src.anki_manager import CardResult
from src.note_templates import CSS
from src.panel_detector import PanelDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Basic HTML wrapper for the flashcards
HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Manga Agent Test Output</title>
<style>
body {{
    background-color: #e0e0e0;
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    padding: 20px;
    justify-content: center;
}}
.flashcard-container {{
    width: 400px;
    background-color: white;
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}}
.card-side {{
    padding: 20px;
    flex-grow: 1;
}}
.card-front {{
    border-bottom: 2px dashed #ccc;
}}
.card-back {{
    background-color: #f9f9f9;
}}
/* Anki CSS */
{anki_css}
</style>
</head>
<body>
"""

HTML_FOOTER = """\
</body>
</html>
"""


class MockAnkiManager:
    def __init__(self, output_html_path: Path):
        self.output_html_path = output_html_path

    def create_kanji_card(self, kanji: str, reading: str, meaning: str, tags: list[str] | None = None) -> CardResult:
        logger.info(f"[MockAnkiManager] create_kanji_card: {kanji}")
        
        front_html = f'<div class="card"><div class="kanji">{kanji}</div></div>'
        back_html = f'<div class="card"><div class="kanji">{kanji}</div><hr id="answer"><div class="reading">{reading}</div><div class="meaning">{meaning}</div></div>'
        
        self._append_card_to_html(front_html, back_html)
        return CardResult(note_id=1, front=kanji, deck="Mock Kanji")

    def create_manga_card(self, word: str, sentence: str, translation: str, image_data: bytes | None = None, reading: str = "", audio_data: bytes | None = None, tags: list[str] | None = None) -> CardResult:
        logger.info(f"[MockAnkiManager] create_manga_card: {word} (reading={reading})")

        # Compress and format the image if provided
        img_html = ""
        if image_data:
            img = Image.open(io.BytesIO(image_data))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            max_dim = 1024
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=80)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            img_html = f'<div class="manga-image"><img src="data:image/webp;base64,{img_b64}"></div>'

        reading_html = f'<div class="reading">{reading}</div>' if reading else ""
        audio_html = ""
        if audio_data:
            audio_b64 = base64.b64encode(audio_data).decode()
            audio_html = f'<div class="audio"><audio controls src="data:audio/wav;base64,{audio_b64}"></audio></div>'

        front_html = f'<div class="card">{img_html}<div class="sentence">{sentence}</div></div>'
        back_html = f'<div class="card">{img_html}<div class="sentence">{sentence}</div><hr id="answer">{reading_html}<div class="translation">{translation}</div>{audio_html}</div>'

        self._append_card_to_html(front_html, back_html)
        return CardResult(note_id=2, front=word, deck="Mock Manga")

    def list_decks(self) -> list[dict]:
        return [{"id": 1, "name": "Mock Deck", "note_count": 0}]

    def search_notes(self, query: str) -> list[dict]:
        return []

    def _append_card_to_html(self, front_html: str, back_html: str):
        card_html = f"""
<div class="flashcard-container">
    <div class="card-side card-front">
        <h3 style="margin-top: 0; color: #666; font-size: 14px; text-align: center;">Front</h3>
        {front_html}
    </div>
    <div class="card-side card-back">
        <h3 style="margin-top: 0; color: #666; font-size: 14px; text-align: center;">Back</h3>
        {back_html}
    </div>
</div>
"""
        with open(self.output_html_path, "a", encoding="utf-8") as f:
            f.write(card_html)


async def main():
    root_dir = Path(__file__).parent
    image_dir = root_dir / "test_manga_images"
    output_html_path = root_dir / "test_output.html"

    # Initialize HTML file
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.format(anki_css=CSS))

    manager = MockAnkiManager(output_html_path)

    logger.info("Building agent...")
    run_agent = build_agent(manager)

    logger.info("Initialising panel detector...")
    detector = PanelDetector(device=settings.panel_model_device)

    # Use the system prompt logic mentioned by user directly by just asking the agent to do its job
    prompt = "Extract vocabulary from this manga page and create cards."

    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.webp"))
    if not images:
        logger.error(f"No images found in {image_dir}")
        return

    for img_path in images:
        logger.info(f"Processing image {img_path.name}...")
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        try:
            page_analysis = detector.detect(image_bytes)
            logger.info(f"Detected {len(page_analysis.panels)} panels in {img_path.name}")
            result = await run_agent(prompt, image_bytes, page_analysis)
            logger.info(f"Agent response for {img_path.name}: {result.text}")

            # Auto-accept all proposed cards (create them via the mock manager)
            if result.pending_cards:
                logger.info(f"Auto-accepting {len(result.pending_cards)} proposed cards")
                for card in result.pending_cards:
                    # Generate TTS for manga cards (same as bot.py does)
                    audio_data = None
                    if card.card_type == "manga" and card.sentence:
                        try:
                            import re
                            from src.tts import generate_tts
                            plain = re.sub(r"<[^>]+>", "", card.sentence)
                            audio_data = generate_tts(plain)
                            logger.info(f"  TTS generated: {len(audio_data)} bytes")
                        except Exception as e:
                            logger.warning(f"  TTS failed for '{card.word}': {e}")
                    manager.create_manga_card(
                        word=card.word, sentence=card.sentence,
                        translation=card.translation,
                        image_data=card.image_data, reading=card.reading,
                        audio_data=audio_data, tags=card.tags,
                    )
        except Exception as e:
            logger.exception(f"Error processing {img_path.name}: {e}")

    # Close HTML file
    with open(output_html_path, "a", encoding="utf-8") as f:
        f.write(HTML_FOOTER)

    logger.info(f"Done. Open {output_html_path} in your browser to view the generated flashcards.")

if __name__ == "__main__":
    asyncio.run(main())
