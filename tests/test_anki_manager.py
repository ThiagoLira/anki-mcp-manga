import pytest

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
        assert field_names == ["Word", "Translation"]
        assert len(nt["tmpls"]) == 1
        assert "{{Word}}" in nt["tmpls"][0]["qfmt"]
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
            word="規則", translation="Rules — 'You must follow the rules.'"
        )
        assert result.front == "規則"
        assert result.note_id > 0

    def test_fields_stored(self, manager):
        result = manager.create_manga_card(
            word="才能", translation="Talent — 'He has incredible talent.'"
        )
        note = manager.col.get_note(result.note_id)
        assert note["Word"] == "才能"
        assert note["Translation"] == "Talent — 'He has incredible talent.'"

    def test_with_tags(self, manager):
        result = manager.create_manga_card(
            word="一撃", translation="One strike — 'He defeated him in one strike.'",
            tags=["one-punch-man"],
        )
        note = manager.col.get_note(result.note_id)
        assert "one-punch-man" in note.tags

    def test_one_card_per_note(self, manager):
        result = manager.create_manga_card(
            word="目立つ", translation="To stand out"
        )
        note = manager.col.get_note(result.note_id)
        assert len(note.cards()) == 1


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
        manager.create_manga_card(word="炎", translation="Flame")
        stats = manager.get_stats()
        assert stats["total_notes"] >= 2
        assert stats["total_cards"] >= 2
