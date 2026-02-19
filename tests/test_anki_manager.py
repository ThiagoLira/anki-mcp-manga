import io

import pytest
from PIL import Image

from src.anki_manager import AnkiManager
from src.note_templates import (
    KANJI_NOTETYPE,
    MANGA_NOTETYPE,
    ensure_kanji_notetype,
    ensure_manga_notetype,
)


@pytest.fixture
def col_path(tmp_path):
    return str(tmp_path / "test.anki2")


@pytest.fixture
def manager(col_path):
    m = AnkiManager(collection_path=col_path)
    yield m
    m.close()


def _make_test_png(width: int = 200, height: int = 100) -> bytes:
    """Generate a small test PNG image."""
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestNoteTemplates:
    def test_ensure_kanji_notetype(self, manager):
        nt = ensure_kanji_notetype(manager.col)
        assert nt["name"] == KANJI_NOTETYPE
        field_names = [f["name"] for f in nt["flds"]]
        assert field_names == ["Kanji", "Reading", "Meaning"]
        assert len(nt["tmpls"]) == 1
        assert "{{Kanji}}" in nt["tmpls"][0]["qfmt"]
        assert "{{Reading}}" in nt["tmpls"][0]["afmt"]

    def test_ensure_manga_notetype(self, manager):
        nt = ensure_manga_notetype(manager.col)
        assert nt["name"] == MANGA_NOTETYPE
        field_names = [f["name"] for f in nt["flds"]]
        assert field_names == ["Word", "Sentence", "Image", "Translation"]
        assert len(nt["tmpls"]) == 1
        assert "{{Sentence}}" in nt["tmpls"][0]["qfmt"]
        assert "{{Translation}}" in nt["tmpls"][0]["afmt"]

    def test_notetypes_idempotent(self, manager):
        nt1 = ensure_kanji_notetype(manager.col)
        nt2 = ensure_kanji_notetype(manager.col)
        assert nt1["id"] == nt2["id"]


class TestKanjiCards:
    def test_create_basic(self, manager):
        result = manager.create_kanji_card(
            kanji="食べる", reading="たべる", meaning="to eat"
        )
        assert result.front == "食べる"
        assert result.note_id > 0

    def test_fields_stored(self, manager):
        result = manager.create_kanji_card(
            kanji="走る", reading="はしる", meaning="to run"
        )
        note = manager.col.get_note(result.note_id)
        assert note["Kanji"] == "走る"
        assert note["Reading"] == "はしる"
        assert note["Meaning"] == "to run"

    def test_with_tags(self, manager):
        result = manager.create_kanji_card(
            kanji="飲む", reading="のむ", meaning="to drink", tags=["verb", "N5"]
        )
        note = manager.col.get_note(result.note_id)
        assert "verb" in note.tags
        assert "N5" in note.tags

    def test_one_card_per_note(self, manager):
        result = manager.create_kanji_card(
            kanji="山", reading="やま", meaning="mountain"
        )
        note = manager.col.get_note(result.note_id)
        assert len(note.cards()) == 1


class TestMangaCards:
    def test_create_basic(self, manager):
        result = manager.create_manga_card(
            word="規則",
            sentence="<b>規則</b>を守れ",
            translation="Follow the <b>rules</b>",
        )
        assert result.front == "規則"
        assert result.note_id > 0

    def test_fields_stored(self, manager):
        result = manager.create_manga_card(
            word="才能",
            sentence="彼は信じられない<b>才能</b>を持っている",
            translation="He has incredible <b>talent</b>",
        )
        note = manager.col.get_note(result.note_id)
        assert note["Word"] == "才能"
        assert "<b>才能</b>" in note["Sentence"]
        assert "<b>talent</b>" in note["Translation"]

    def test_with_tags(self, manager):
        result = manager.create_manga_card(
            word="一撃",
            sentence="<b>一撃</b>で倒した",
            translation="Defeated in <b>one strike</b>",
            tags=["one-punch-man"],
        )
        note = manager.col.get_note(result.note_id)
        assert "one-punch-man" in note.tags

    def test_one_card_per_note(self, manager):
        result = manager.create_manga_card(
            word="目立つ",
            sentence="彼は<b>目立つ</b>存在だ",
            translation="He is a presence that <b>stands out</b>",
        )
        note = manager.col.get_note(result.note_id)
        assert len(note.cards()) == 1

    def test_create_with_image(self, manager):
        image_data = _make_test_png()
        result = manager.create_manga_card(
            word="戦う",
            sentence="<b>戦う</b>しかない",
            translation="There's no choice but to <b>fight</b>",
            image_data=image_data,
        )
        note = manager.col.get_note(result.note_id)
        assert '<img src="manga_' in note["Image"]
        assert note["Image"].endswith('.webp">')

    def test_create_with_large_image_resized(self, manager):
        image_data = _make_test_png(width=2048, height=1536)
        result = manager.create_manga_card(
            word="巨大",
            sentence="<b>巨大</b>な敵が現れた",
            translation="A <b>huge</b> enemy appeared",
            image_data=image_data,
        )
        note = manager.col.get_note(result.note_id)
        assert '<img src="manga_' in note["Image"]

    def test_create_without_image(self, manager):
        result = manager.create_manga_card(
            word="静か",
            sentence="<b>静か</b>にしろ",
            translation="Be <b>quiet</b>",
        )
        note = manager.col.get_note(result.note_id)
        assert note["Image"] == ""


class TestCollectionOps:
    def test_list_decks(self, manager):
        manager.create_kanji_card(kanji="犬", reading="いぬ", meaning="dog")
        decks = manager.list_decks()
        names = [d["name"] for d in decks]
        assert "Japones KANJI" in names

    def test_search_notes(self, manager):
        manager.create_kanji_card(kanji="水", reading="みず", meaning="water")
        results = manager.search_notes("water")
        assert len(results) >= 1
        assert results[0]["fields"]["Kanji"] == "水"

    def test_get_stats(self, manager):
        manager.create_kanji_card(kanji="火", reading="ひ", meaning="fire")
        manager.create_manga_card(word="炎", sentence="<b>炎</b>が燃えている", translation="The <b>flames</b> are burning")
        stats = manager.get_stats()
        assert stats["total_notes"] >= 2
        assert stats["total_cards"] >= 2
