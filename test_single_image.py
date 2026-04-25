"""Quick test: run a single image through the pipeline and print results."""
import asyncio
import logging

from src.config import settings
from src.agent import CardAgent
from src.panel_detector import PanelDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    with open("test_single.jpg", "rb") as f:
        image_bytes = f.read()

    logger.info("Detecting panels...")
    detector = PanelDetector(device=settings.panel_model_device)
    page_analysis = detector.detect(image_bytes)
    logger.info("Detected %d panels", len(page_analysis.panels))

    agent = CardAgent()
    result = await agent.process_image(
        "Extract vocabulary from this manga page and create cards.",
        image_bytes,
        page_analysis,
    )

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(result.text)
    print("=" * 60)
    print(f"\nPROPOSED CARDS ({len(result.pending_cards)}):")
    for i, card in enumerate(result.pending_cards):
        print(f"  {i+1}. {card.word} ({card.reading})")
        print(f"     sentence: {card.sentence}")
        print(f"     translation: {card.translation}")
        print(f"     image: {'panel' if card.image_data else 'none'} ({len(card.image_data)} bytes)" if card.image_data else "     image: none")
        print()


if __name__ == "__main__":
    asyncio.run(main())
