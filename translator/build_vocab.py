"""
translator/build_vocab.py
=========================
Pre-populate the Loga dictionary by scanning the English Wikipedia corpus
for frequent content words and assigning them stable Loga roots.

Outputs data/dictionary.json with:
- Proper nouns (from seed list + extracted from corpus)
- Common nouns mapped to unused 2-char lowercase roots
- Common verbs mapped to unused 2-char uppercase roots
- Frequent compounds for less common terms

Usage:
    python -m translator.build_vocab \
        --input ../conlang-experiment/data/raw/english/simplewiki-articles.jsonl \
        --output data/dictionary.json \
        --max-nouns 300 --max-verbs 200
"""

from __future__ import annotations

import json
import logging
import re
import string
from collections import Counter
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Already-assigned roots from the Loga spec (cannot be reused)
# ---------------------------------------------------------------------------

CORE_NOUNS = {
    "mi": "I/me", "tu": "you", "si": "he/she/it", "ma": "we", "na": "you(pl)", "sa": "they",
    "ka": "person", "ku": "water", "ge": "fire", "to": "time", "se": "place",
    "li": "life", "bo": "city", "pa": "idea", "da": "thing", "ne": "name",
    "la": "land", "ha": "leader", "re": "rule", "go": "road", "no": "knowledge",
    "fa": "family", "wa": "conflict", "pe": "food", "de": "death", "gi": "start",
    "ta": "end", "nu": "number", "su": "part", "yu": "purpose", "ro": "work",
    "lo": "location", "ve": "event", "fi": "feeling", "wi": "what", "bi": "big",
    "mo": "many", "zo": "zero", "ra": "animal", "pi": "plant", "co": "country",
}

CORE_VERBS = {
    "Be": "be", "Go": "go", "Se": "see", "Sa": "say", "Ma": "make",
    "Gi": "give", "Ta": "take", "Ea": "eat", "Sl": "sleep", "Th": "think",
    "Kn": "know", "Wa": "walk", "Si": "sit", "Ha": "have", "Us": "use",
    "Li": "live", "Di": "die", "Ca": "call", "Co": "come", "Le": "leave",
    "Fi": "find", "St": "start", "En": "end", "Ch": "change", "Ru": "rule",
    "Fo": "follow", "Wo": "work", "Fe": "feel", "Wi": "want", "Re": "return",
    "Bu": "build", "Br": "bring", "Sp": "speak", "Tr": "travel",
}

# English stop words — never assign roots to these
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "this", "that",
    "these", "those", "what", "which", "who", "whom", "whose",
    "and", "but", "or", "nor", "not", "no", "so", "if", "then", "than",
    "as", "at", "by", "for", "from", "in", "into", "of", "on", "to",
    "with", "about", "after", "before", "between", "during", "through",
    "above", "below", "up", "down", "out", "off", "over", "under",
    "again", "further", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "very", "just", "also", "now",
    "new", "first", "last", "long", "great", "little", "just", "like",
    "many", "much", "still", "even", "back", "well", "also", "however",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "called", "used", "known", "made", "became", "became", "found", "part",
    "people", "year", "years", "time", "world", "city", "country", "state",
    "also", "often", "later", "since", "while", "because", "although",
    "any", "another", "between", "until", "before", "after", "against",
    "during", "without", "within", "along", "including", "following",
}

# Common English verbs likely to appear in Wikipedia
VERB_INDICATORS = {
    "run", "walk", "talk", "write", "read", "play", "fight", "win", "lose",
    "grow", "fall", "rise", "move", "turn", "keep", "hold", "put", "set",
    "show", "tell", "ask", "try", "begin", "help", "open", "close",
    "stop", "cut", "pay", "meet", "stand", "run", "lead", "learn", "teach",
    "carry", "break", "reach", "kill", "remain", "believe", "provide",
    "happen", "include", "continue", "allow", "serve", "appear", "cover",
    "produce", "develop", "report", "remember", "consider", "create",
    "form", "include", "describe", "receive", "support", "send", "expect",
    "build", "stay", "sing", "dance", "swim", "fly", "drive", "ride",
    "wear", "hang", "spread", "strike", "contain", "involve", "require",
    "suggest", "control", "manage", "destroy", "discover", "release",
    "replace", "protect", "publish", "add", "join", "accept", "agree",
    "achieve", "represent", "establish", "connect", "separate",
    "born", "die", "live", "love", "hate", "want", "need",
}


def extract_word_frequencies(input_path: Path, max_articles: int = 0) -> Counter:
    """Count word frequencies across the corpus."""
    freq = Counter()
    count = 0
    with open(input_path) as f:
        for line in f:
            try:
                article = json.loads(line)
                text = article.get("text", "")
            except (json.JSONDecodeError, KeyError):
                continue
            # Tokenize: lowercase, alpha only
            words = re.findall(r"[a-z]+", text.lower())
            freq.update(words)
            count += 1
            if max_articles > 0 and count >= max_articles:
                break
    log.info("Scanned %d articles, %d unique words", count, len(freq))
    return freq


def extract_proper_nouns(input_path: Path, max_articles: int = 0, min_count: int = 5) -> Counter:
    """Extract capitalized multi-word and single-word proper nouns."""
    freq = Counter()
    count = 0
    with open(input_path) as f:
        for line in f:
            try:
                article = json.loads(line)
                text = article.get("text", "")
            except (json.JSONDecodeError, KeyError):
                continue
            # Multi-word proper nouns (e.g., "United States", "New York")
            for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
                freq[match.group()] += 1
            # Single-word proper nouns (capitalized, not at sentence start)
            for match in re.finditer(r"(?<=[a-z]\s)([A-Z][a-z]{2,})\b", text):
                freq[match.group()] += 1
            count += 1
            if max_articles > 0 and count >= max_articles:
                break
    # Filter low-frequency
    return Counter({k: v for k, v in freq.items() if v >= min_count})


def generate_noun_roots(used: set[str]) -> list[str]:
    """Generate available 2-char noun roots (lowercase first char)."""
    second_chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
    roots = []
    for c1 in string.ascii_lowercase:
        for c2 in second_chars:
            root = c1 + c2
            if root not in used:
                roots.append(root)
    return roots


def generate_verb_roots(used: set[str]) -> list[str]:
    """Generate available 2-char verb roots (uppercase first char)."""
    second_chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
    roots = []
    for c1 in string.ascii_uppercase:
        for c2 in second_chars:
            root = c1 + c2
            if root not in used:
                roots.append(root)
    return roots


def generate_proper_noun_code(term: str, used: set[str]) -> str | None:
    """Generate a unique 2-char proper noun code (uppercase first char)."""
    # Try natural abbreviation first
    words = term.split()
    if len(words) >= 2:
        # Try initials: "United States" → "US"
        code = words[0][0].upper() + words[1][0].upper()
        if code not in used:
            return code
        # Try first + second char of second word
        code = words[0][0].upper() + words[1][1] if len(words[1]) > 1 else None
        if code and code not in used:
            return code
    # Single word: try first two chars
    code = term[0].upper() + term[1].lower() if len(term) > 1 else None
    if code and code not in used:
        return code
    # Try first + last char
    code = term[0].upper() + term[-1].lower() if len(term) > 1 else None
    if code and code not in used:
        return code
    # Brute force: first char + any available second char
    second_chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
    for c2 in second_chars:
        code = term[0].upper() + c2
        if code not in used:
            return code
    return None


def classify_word(word: str, freq: Counter) -> str:
    """Heuristic: is this word more likely a noun or verb in Wikipedia?"""
    if word in VERB_INDICATORS:
        return "verb"
    # Words ending in common verb suffixes
    if word.endswith(("ed", "ing", "ize", "ify", "ate")):
        return "verb"
    # Words ending in common noun suffixes
    if word.endswith(("tion", "sion", "ment", "ness", "ity", "ance", "ence",
                       "ism", "ist", "ology", "er", "or", "al")):
        return "noun"
    # Default: noun (more common in encyclopedic text)
    return "noun"


def build_vocabulary(
    input_path: Path,
    output_path: Path,
    max_nouns: int,
    max_verbs: int,
    max_proper: int,
    max_articles: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing dictionary if present
    existing: dict[str, str] = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f).get("entries", {})
        log.info("Existing dictionary: %d entries", len(existing))

    # Track all used codes
    used_codes: set[str] = set()
    used_codes.update(CORE_NOUNS.keys())
    used_codes.update(CORE_VERBS.keys())
    used_codes.update(existing.values())

    entries: dict[str, str] = dict(existing)

    # Import seed dictionary from translate.py
    from translator.translate import SEED_DICTIONARY
    for term, code in SEED_DICTIONARY.items():
        if term not in entries:
            entries[term] = code
            used_codes.add(code)
    log.info("After seed: %d entries", len(entries))

    # Extract frequencies
    log.info("Extracting word frequencies...")
    freq = extract_word_frequencies(input_path, max_articles)

    # Extract proper nouns
    log.info("Extracting proper nouns...")
    proper_freq = extract_proper_nouns(input_path, max_articles)
    proper_sorted = proper_freq.most_common(max_proper)

    # Assign proper noun codes
    proper_added = 0
    for term, count in proper_sorted:
        if term in entries:
            continue
        code = generate_proper_noun_code(term, used_codes)
        if code:
            entries[term] = code
            used_codes.add(code)
            proper_added += 1
    log.info("Proper nouns added: %d (from %d candidates)", proper_added, len(proper_sorted))

    # Filter content words
    content_words = {
        w: c for w, c in freq.items()
        if w not in STOP_WORDS
        and len(w) >= 3
        and c >= 10
        and w not in {v.lower() for v in entries}
    }

    # Classify and sort
    nouns = [(w, c) for w, c in content_words.items() if classify_word(w, freq) == "noun"]
    verbs = [(w, c) for w, c in content_words.items() if classify_word(w, freq) == "verb"]
    nouns.sort(key=lambda x: -x[1])
    verbs.sort(key=lambda x: -x[1])

    log.info("Content word candidates: %d nouns, %d verbs", len(nouns), len(verbs))

    # Assign noun roots
    available_noun_roots = generate_noun_roots(used_codes)
    noun_added = 0
    for word, count in nouns[:max_nouns]:
        if not available_noun_roots:
            break
        if word in entries or word in CORE_NOUNS.values():
            continue
        root = available_noun_roots.pop(0)
        entries[word] = root
        used_codes.add(root)
        noun_added += 1
    log.info("Noun roots assigned: %d (of %d available)", noun_added, len(available_noun_roots) + noun_added)

    # Assign verb roots
    available_verb_roots = generate_verb_roots(used_codes)
    verb_added = 0
    for word, count in verbs[:max_verbs]:
        if not available_verb_roots:
            break
        if word in entries or word in CORE_VERBS.values():
            continue
        root = available_verb_roots.pop(0)
        entries[word] = root
        used_codes.add(root)
        verb_added += 1
    log.info("Verb roots assigned: %d (of %d available)", verb_added, len(available_verb_roots) + verb_added)

    # Save
    with open(output_path, "w") as f:
        json.dump({"entries": entries}, f, indent=2, ensure_ascii=False)

    log.info(
        "Dictionary saved: %d total entries (%d proper, %d nouns, %d verbs, %d seed/existing)",
        len(entries), proper_added, noun_added, verb_added,
        len(entries) - proper_added - noun_added - verb_added,
    )

    # Print top entries for review
    print("\n--- Top 20 assigned nouns ---")
    noun_entries = [(w, c, entries[w]) for w, c in nouns[:20] if w in entries]
    for word, count, root in noun_entries:
        print(f"  {word:20s} (freq {count:6d}) → {root}")

    print("\n--- Top 20 assigned verbs ---")
    verb_entries = [(w, c, entries[w]) for w, c in verbs[:20] if w in entries]
    for word, count, root in verb_entries:
        print(f"  {word:20s} (freq {count:6d}) → {root}")

    print(f"\n--- Top 20 proper nouns ---")
    for term, count in proper_sorted[:20]:
        code = entries.get(term, "?")
        print(f"  {term:30s} (freq {count:6d}) → {code}")


@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path),
              required=True, help="Path to simplewiki-articles.jsonl")
@click.option("--output", "output_path", type=click.Path(path_type=Path),
              default="data/dictionary.json", show_default=True)
@click.option("--max-nouns", default=300, show_default=True)
@click.option("--max-verbs", default=200, show_default=True)
@click.option("--max-proper", default=500, show_default=True,
              help="Max proper nouns to extract from corpus")
@click.option("--max-articles", default=0, show_default=True,
              help="Limit articles to scan (0 = all)")
def cli(input_path, output_path, max_nouns, max_verbs, max_proper, max_articles):
    """Build Loga vocabulary dictionary from English Wikipedia corpus."""
    build_vocabulary(input_path, output_path, max_nouns, max_verbs, max_proper, max_articles)


if __name__ == "__main__":
    cli()
