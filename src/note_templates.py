from anki.collection import Collection

# --- Shared CSS ---

CSS = """\
.card {
    font-family: "Hiragino Kaku Gothic Pro", "Noto Sans JP", "Meiryo", sans-serif;
    font-size: 24px;
    text-align: center;
    color: #1a1a2e;
    background-color: #f5f5f5;
    padding: 20px;
}
.kanji {
    font-size: 48px;
    font-weight: bold;
    color: #16213e;
    margin: 16px 0;
}
.reading {
    font-size: 28px;
    color: #0f3460;
    margin: 8px 0;
}
.meaning {
    font-size: 22px;
    color: #333;
    margin: 8px 0;
}
.word {
    font-size: 32px;
    font-weight: bold;
    color: #16213e;
    margin: 12px 0;
}
.image img {
    max-width: 100%;
    max-height: 300px;
    border-radius: 8px;
    margin: 12px 0;
}
.translation {
    font-size: 20px;
    color: #333;
    margin: 12px 0;
}
hr#answer {
    border: none;
    border-top: 2px solid #ddd;
    margin: 16px 0;
}
"""

# --- Kanji notetype ---
# Front: kanji/vocab word
# Back: reading + meaning

KANJI_NOTETYPE = "Kanji"
KANJI_FIELDS = ["Kanji", "Reading", "Meaning"]

KANJI_QFMT = '<div class="kanji">{{Kanji}}</div>'
KANJI_AFMT = """\
{{FrontSide}}
<hr id="answer">
<div class="reading">{{Reading}}</div>
<div class="meaning">{{Meaning}}</div>
"""

# --- Manga Vocab notetype ---
# Front: manga panel screenshot + word
# Back: sentence translation

MANGA_NOTETYPE = "Manga Vocab"
MANGA_FIELDS = ["Word", "Image", "Translation"]

MANGA_QFMT = """\
<div class="word">{{Word}}</div>
{{#Image}}<div class="image">{{Image}}</div>{{/Image}}
"""
MANGA_AFMT = """\
{{FrontSide}}
<hr id="answer">
<div class="translation">{{Translation}}</div>
"""


def _ensure(col: Collection, name: str, fields: list[str], qfmt: str, afmt: str) -> dict:
    """Get or create a notetype with a single card template."""
    existing = col.models.by_name(name)
    if existing:
        return existing

    model = col.models.new(name)
    model["css"] = CSS

    for field_name in fields:
        field = col.models.new_field(field_name)
        col.models.add_field(model, field)

    template = col.models.new_template("Card 1")
    template["qfmt"] = qfmt
    template["afmt"] = afmt
    col.models.add_template(model, template)

    col.models.add(model)
    return model


def ensure_kanji_notetype(col: Collection) -> dict:
    return _ensure(col, KANJI_NOTETYPE, KANJI_FIELDS, KANJI_QFMT, KANJI_AFMT)


def ensure_manga_notetype(col: Collection) -> dict:
    return _ensure(col, MANGA_NOTETYPE, MANGA_FIELDS, MANGA_QFMT, MANGA_AFMT)
