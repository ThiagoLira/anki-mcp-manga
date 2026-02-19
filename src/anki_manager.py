import hashlib
import io
import os
from dataclasses import dataclass

from anki.collection import Collection
from PIL import Image

from .config import settings
from .note_templates import ensure_kanji_notetype, ensure_manga_notetype


@dataclass
class CardResult:
    note_id: int
    front: str
    deck: str


class AnkiManager:
    def __init__(self, collection_path: str | None = None):
        self._path = collection_path or settings.collection_path
        self._col: Collection | None = None

    @property
    def col(self) -> Collection:
        if self._col is None:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            self._col = Collection(self._path)
        return self._col

    def close(self):
        if self._col is not None:
            self._col.close()
            self._col = None

    def reopen(self):
        """Close and reopen the collection (needed after full upload/download)."""
        if self._col is not None:
            self._col.close()
            self._col = None
        # Lazy reopen on next .col access

    def create_kanji_card(
        self,
        kanji: str,
        reading: str,
        meaning: str,
        tags: list[str] | None = None,
    ) -> CardResult:
        """Create a kanji/vocab card. Front: kanji. Back: reading + meaning."""
        col = self.col
        notetype = ensure_kanji_notetype(col)
        deck_id = col.decks.id(settings.kanji_deck)

        note = col.new_note(notetype)
        note["Kanji"] = kanji
        note["Reading"] = reading
        note["Meaning"] = meaning

        if tags:
            for tag in tags:
                note.add_tag(tag)

        col.add_note(note, deck_id)
        return CardResult(note_id=note.id, front=kanji, deck=settings.kanji_deck)

    def create_manga_card(
        self,
        word: str,
        sentence: str,
        translation: str,
        image_data: bytes | None = None,
        tags: list[str] | None = None,
    ) -> CardResult:
        """Create a manga vocab card.

        Front: manga image + Japanese sentence (target word bolded).
        Back: full sentence translation (target word bolded).
        """
        col = self.col
        notetype = ensure_manga_notetype(col)
        deck_id = col.decks.id(settings.manga_deck)

        note = col.new_note(notetype)
        note["Word"] = word
        note["Sentence"] = sentence
        note["Translation"] = translation

        if image_data:
            filename = self._process_image(image_data)
            note["Image"] = f'<img src="{filename}">'

        if tags:
            for tag in tags:
                note.add_tag(tag)

        col.add_note(note, deck_id)
        return CardResult(note_id=note.id, front=word, deck=settings.manga_deck)

    def _process_image(self, image_data: bytes) -> str:
        """Resize, convert to WebP, and save to collection media. Returns filename."""
        img = Image.open(io.BytesIO(image_data))

        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        max_dim = 1024
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=80)
        webp_bytes = buf.getvalue()

        image_hash = hashlib.sha256(webp_bytes).hexdigest()[:12]
        filename = f"manga_{image_hash}.webp"
        return self.col.media.write_data(filename, webp_bytes)

    def list_decks(self) -> list[dict]:
        """List all decks with card counts."""
        result = []
        for deck_entry in self.col.decks.all_names_and_ids():
            count = len(self.col.find_notes(f'"deck:{deck_entry.name}"'))
            result.append(
                {"id": deck_entry.id, "name": deck_entry.name, "note_count": count}
            )
        return result

    def search_notes(self, query: str) -> list[dict]:
        """Search notes using Anki search syntax. Returns up to 50 results."""
        note_ids = self.col.find_notes(query)
        results = []
        for nid in note_ids[:50]:
            note = self.col.get_note(nid)
            results.append(
                {
                    "id": note.id,
                    "fields": {k: note[k] for k in note.keys()},
                    "tags": note.tags,
                }
            )
        return results

    def get_stats(self) -> dict:
        """Get collection overview stats."""
        col = self.col
        return {
            "total_notes": col.note_count(),
            "total_cards": col.card_count(),
            "total_decks": len(col.decks.all_names_and_ids()),
            "studied_today": col.studied_today(),
        }
