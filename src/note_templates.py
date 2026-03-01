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
.manga-image img {
    max-width: 100%;
    max-height: 300px;
    border-radius: 8px;
    margin: 12px 0;
}
.sentence {
    font-size: 26px;
    color: #1a1a2e;
    margin: 12px 0;
    line-height: 1.5;
}
.sentence b {
    color: #16213e;
    font-size: 30px;
}
.translation {
    font-size: 20px;
    color: #333;
    margin: 12px 0;
    line-height: 1.4;
}
.translation b {
    color: #0f3460;
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
# Front: manga image + Japanese sentence (target word bolded)
# Back: full sentence translation (target word bolded)

MANGA_NOTETYPE = "Manga Vocab"
MANGA_FIELDS = ["Word", "Sentence", "Image", "Translation", "Reading", "Audio"]

MANGA_QFMT = """\
{{#Image}}<div class="manga-image">{{Image}}</div>{{/Image}}
<div class="sentence">{{Sentence}}</div>
"""
MANGA_AFMT = """\
{{FrontSide}}
<hr id="answer">
{{#Reading}}<div class="reading">{{Reading}}</div>{{/Reading}}
<div class="translation">{{Translation}}</div>
{{#Audio}}<div class="audio">{{Audio}}</div>{{/Audio}}
"""


def _ensure(col: Collection, name: str, fields: list[str], qfmt: str, afmt: str) -> dict:
    """Get or create a notetype with a single card template.

    If the notetype already exists, adds any missing fields and updates
    the card template formats (migration for schema changes).
    """
    existing = col.models.by_name(name)
    if existing:
        # Add any missing fields (migration)
        existing_names = {f["name"] for f in existing["flds"]}
        changed = False
        for field_name in fields:
            if field_name not in existing_names:
                new_field = col.models.new_field(field_name)
                col.models.add_field(existing, new_field)
                changed = True
        # Update template formats if they changed
        tmpl = existing["tmpls"][0]
        if tmpl["qfmt"] != qfmt or tmpl["afmt"] != afmt:
            tmpl["qfmt"] = qfmt
            tmpl["afmt"] = afmt
            changed = True
        if changed:
            col.models.update_dict(existing)
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
