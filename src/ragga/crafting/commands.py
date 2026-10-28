from functools import cache

import spacy
from spacy.language import Language

SPACY_MODEL = "en_core_web_sm"


@cache
def get_nlp() -> Language:
    """Load the spaCy pipeline once, on first use (it's installed as a package dependency)."""
    try:
        return spacy.load(SPACY_MODEL)
    except OSError as e:
        msg = f"spaCy model '{SPACY_MODEL}' is not installed, reinstall ragga or install the model manually."
        raise RuntimeError(msg) from e


def extract_keywords(
    text: str, known_locations: set[str] | None = None
) -> tuple[str | None, str | None, str | None, bool]:
    """
    Extracts the command, what and where from a text.
    :param text: a text
    :return: a tuple of command, what and where
    """
    doc = get_nlp()(text)
    what_pos = set({"NOUN", "PROPN", "ADP"})
    exclude_dep = set({"poss", "prep"})
    where_dep = set({"dobj", "pobj", "compound", "amod", "nsubj"})
    command = None
    what = None
    what_root = None
    where = None
    known = False
    for s in doc.sents:
        command = s.root.lemma_
        break
    for token in doc:
        if token.dep_ == "punct":
            continue
        if token.head.lemma_ == command:
            what_root = token
            what = " ".join([t.text for t in token.subtree if t.dep_ not in exclude_dep])
        if token.pos_ in what_pos:
            if where is None:
                if token.dep_ in where_dep and (token.head.lemma_ == command or token.head == what_root):
                    where = " ".join([t.text for t in token.subtree if t.dep_ not in exclude_dep])

    if command is not None:
        command = command.lower()
    if where == what:
        where = None
    if where is not None:
        where = where.lower()
    if what is not None:
        what = what.lower()

    # switcheroo
    if known_locations is not None:
        known_locations = set({loc.lower() for loc in known_locations})
        if where is not None:
            if where in known_locations:
                known = True
            elif what in known_locations:
                what, where = where, what
                known = True

    return command, where, what, known


if __name__ == "__main__":
    print("========================")  # noqa: T201
    print("Testing extract_keywords")  # noqa: T201
    print("========================")  # noqa: T201
    texts = [
        "what have I written about sonification?",
        "what do my notes say about sonification?",
        "tell me about sonification",
        "what is sonification?",
        "for my IMC work, what are my hours?",
        "summarise my IMC work",
        "summarise my work for IMC",
        "summarise writings about sonification",
        "summarise Natural Language Processing notes",
        "summarise natural language processing",
        "summarise advanced cognitive neuroscience",
        "summarise philosophy of cognitive science",
        "summarise my notes on philosophy of cognitive science",
    ]

    for text in texts:
        print()  # noqa: T201
        print(f"text:    {text}")  # noqa: T201
        known_locations = set({"notes", "note", "writing", "writings"})
        cmd, where, what, known = extract_keywords(text, known_locations)
        print(f"command: {cmd}")  # noqa: T201
        print(f"where:   {where}")  # noqa: T201
        print(f"what:    {what}")  # noqa: T201
        print(f"known:   {known}")  # noqa: T201
