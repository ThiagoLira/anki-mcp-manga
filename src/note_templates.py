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
# Two cards per note:
#   Reading  — front: sentence (text), back: image + reading + audio + translation
#   Listening — front: audio + word, back: sentence + image + translation

MANGA_NOTETYPE = "Manga Vocab"
MANGA_FIELDS = ["Word", "Sentence", "Image", "Translation", "Reading", "Audio"]

MANGA_TEMPLATES = [
    {
        "name": "Reading",
        "qfmt": '<div class="sentence">{{Sentence}}</div>',
        "afmt": """\
{{FrontSide}}
<hr id="answer">
{{#Image}}<div class="manga-image">{{Image}}</div>{{/Image}}
{{#Reading}}<div class="reading">{{Reading}}</div>{{/Reading}}
<div class="translation">{{Translation}}</div>
{{#Audio}}<div class="audio">{{Audio}}</div>{{/Audio}}""",
    },
    {
        "name": "Listening",
        "qfmt": """\
{{#Audio}}<div class="audio">{{Audio}}</div>{{/Audio}}
<div class="kanji">{{Word}}</div>""",
        "afmt": """\
{{FrontSide}}
<hr id="answer">
{{#Image}}<div class="manga-image">{{Image}}</div>{{/Image}}
<div class="sentence">{{Sentence}}</div>
<div class="translation">{{Translation}}</div>""",
    },
]


def _ensure(
    col: Collection,
    name: str,
    fields: list[str],
    templates: list[dict[str, str]],
) -> dict:
    """Get or create a notetype with one or more card templates.

    Each entry in *templates* is ``{"name": ..., "qfmt": ..., "afmt": ...}``.

    If the notetype already exists, adds any missing fields and syncs
    templates (add new ones, update changed formats).
    """
    existing = col.models.by_name(name)
    if existing:
        # Add any missing fields
        existing_names = {f["name"] for f in existing["flds"]}
        changed = False
        for field_name in fields:
            if field_name not in existing_names:
                new_field = col.models.new_field(field_name)
                col.models.add_field(existing, new_field)
                changed = True

        # Rename legacy templates before syncing (e.g. "Card 1" → "Reading")
        existing_tmpls = {t["name"]: t for t in existing["tmpls"]}
        desired_names = {t["name"] for t in templates}
        if len(templates) > 0 and len(existing["tmpls"]) == 1:
            old_tmpl = existing["tmpls"][0]
            if old_tmpl["name"] not in desired_names:
                old_tmpl["name"] = templates[0]["name"]
                existing_tmpls = {old_tmpl["name"]: old_tmpl}
                changed = True

        # Sync templates: update existing, add missing
        for tdef in templates:
            if tdef["name"] in existing_tmpls:
                tmpl = existing_tmpls[tdef["name"]]
                if tmpl["qfmt"] != tdef["qfmt"] or tmpl["afmt"] != tdef["afmt"]:
                    tmpl["qfmt"] = tdef["qfmt"]
                    tmpl["afmt"] = tdef["afmt"]
                    changed = True
            else:
                new_tmpl = col.models.new_template(tdef["name"])
                new_tmpl["qfmt"] = tdef["qfmt"]
                new_tmpl["afmt"] = tdef["afmt"]
                col.models.add_template(existing, new_tmpl)
                changed = True

        if changed:
            col.models.update_dict(existing)
        return existing

    model = col.models.new(name)
    model["css"] = CSS

    for field_name in fields:
        fld = col.models.new_field(field_name)
        col.models.add_field(model, fld)

    for tdef in templates:
        tmpl = col.models.new_template(tdef["name"])
        tmpl["qfmt"] = tdef["qfmt"]
        tmpl["afmt"] = tdef["afmt"]
        col.models.add_template(model, tmpl)

    col.models.add(model)
    return model


def ensure_kanji_notetype(col: Collection) -> dict:
    return _ensure(col, KANJI_NOTETYPE, KANJI_FIELDS, [
        {"name": "Card 1", "qfmt": KANJI_QFMT, "afmt": KANJI_AFMT},
    ])


def ensure_manga_notetype(col: Collection) -> dict:
    return _ensure(col, MANGA_NOTETYPE, MANGA_FIELDS, MANGA_TEMPLATES)
