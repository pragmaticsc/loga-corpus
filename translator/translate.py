"""
translator/translate.py
=======================
English → Loga translation using a local LLM (Gemma via mlx-lm).

Reads Simple English Wikipedia articles from a JSONL file, translates each
article to Loga, and writes output as JSONL. Resumable — skips articles
already present in the output file.

A persistent dictionary (data/dictionary.json) grows during translation:
- Before each article, the current dictionary is injected into the prompt
- After each article, new codes found in the output are extracted and added
- This ensures the same entity/concept always gets the same Loga code

Usage:
    python -m translator.translate \
        --input ../conlang-experiment/data/raw/english/simplewiki-articles.jsonl \
        --output data/loga-articles.jsonl \
        --model mlx-community/gemma-4-31b-it-8bit

Requirements:
    pip install -e .
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import click
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loga grammar reference — injected as system/context for every translation
# ---------------------------------------------------------------------------

GRAMMAR_PREAMBLE = """\
You are a translator from English to Loga v0.2, a constructed language designed for
LLM tokenization efficiency. Follow these rules exactly.

WORD STRUCTURE (always exactly 3 characters, space-delimited):
  Nouns:  [C₁][C₂][CASE]   — 2-char root + 1 case suffix
  Verbs:  [C₁][C₂][TENSE]  — 2-char root + 1 tense marker

GRAMMATICAL CLASS (determined by first character C₁):
  Lowercase a-z → NOUN root
  Uppercase A-Z → VERB root
  Digit 0-9     → NUMBER literal (no suffix needed)

CASE SUFFIXES (third character of nouns — from the !-/ range):
  !  nominative   — subject of verb
  "  accusative   — direct object
  #  genitive     — possessor / of-relation
  $  dative       — indirect object / recipient
  %  locative     — location (at / on / in)
  &  lative       — direction (toward / to)
  '  ablative     — source (from)
  (  instrumental — means / tool
  )  comitative   — accompaniment (with)
  *  causative    — cause / because of
  +  benefactive  — for the benefit of
  ,  comparative  — more than
  -  adjectival   — modifies preceding noun (adjective)
  .  adverbial    — modifies verb
  /  vocative     — direct address

TENSE / ASPECT MARKERS (third character of verbs — from the :-@ range):
  :  present
  ;  past
  <  future
  =  perfective (completed action)
  >  imperfective (ongoing / habitual)
  ?  interrogative (question)
  @  conditional

SYNTAX: strict SOV (Subject-Object-Verb). No exceptions. No movement.
  [SUBJ!] [OBJ"] [VERB:] .   (declarative: standalone "." after verb, space-delimited)
  [SUBJ!] [OBJ"] [VERB?]     (yes/no question: no period; "?" is verb tense marker)
  [wi!]   [OBJ"] [VERB:] .   (wh-question: use "wi" pronoun + normal tense + ".")

PARTICLES (single characters; precede the word they modify):
  [  all, every
  \\  some, a few
  ]  one (indefinite)
  ^  none, zero
  _  negation (directly before verb or noun)
  `  subordinate clause introducer ("that")

COMPOUNDS (join two roots with {):
  ku{Ma  = water+make = irrigate
  ge{se  = fire+place = volcano
  pa{ka  = idea+person = philosopher

CORE VOCABULARY:

Pronouns (noun roots):
  mi=I/me  tu=you(sg)  si=he/she/it  ma=we  na=you(pl)  sa=they

Core nouns (all lowercase first char):
  ka=person/human  ku=water    ge=fire     to=time     se=place/world
  li=life          bo=city     pa=idea     da=thing    ne=name/word
  la=land/ground   ha=leader   re=rule/law go=road     no=knowledge
  fa=group/family  wa=conflict pe=food     de=death    gi=start
  ta=end           nu=number   su=part     yu=purpose  ro=work
  lo=location      ve=event    fi=feeling  wi=what/who/which(interrogative)  bi=size/big
  mo=amount/many   zo=zero     ra=animal   pi=plant    co=country

Core verbs (all uppercase first char):
  Be=be(copula)  Go=go/move  Se=see      Sa=say/speak  Ma=make/create
  Gi=give        Ta=take     Ea=eat      Sl=sleep       Th=think
  Kn=know        Wa=walk     Si=sit      Ha=have        Us=use
  Li=live        Di=die      Ca=call     Co=come        Le=leave
  Fi=find        St=start    En=end      Ch=change      Ru=rule
  Fo=follow      Wo=work     Fe=feel     Wi=want        Re=return
  Bu=build       Br=bring    Sp=speak    Tr=travel

Numbers: use digit literals directly. Compound: 42, 100, 1945.

TRANSLATION RULES:
1. Translate MEANING, not word-for-word. Choose the closest Loga root.
2. For new concepts: use a compound (root{root) or abbreviate to 2 chars.
3. Keep sentences short. Split long English sentences into multiple Loga sentences.
4. Preserve paragraph structure (blank lines between paragraphs).
5. Output ONLY Loga text. NO English. NO explanation. NO commentary.
6. Every noun must have exactly one case suffix. Every verb must have exactly one tense marker.
7. Sentence-final period "." is a standalone space-delimited token after the verb: e.g. "Go; ." not "Go;."
"""


# ---------------------------------------------------------------------------
# Persistent dictionary
# ---------------------------------------------------------------------------

# Codes reserved by core vocabulary — proper nouns must not reuse these
RESERVED_CODES = {
    # Core verb roots
    "Be", "Go", "Se", "Sa", "Ma", "Gi", "Ta", "Ea", "Sl", "Th",
    "Kn", "Wa", "Si", "Ha", "Us", "Li", "Di", "Ca", "Co", "Le",
    "Fi", "St", "En", "Ch", "Ru", "Fo", "Wo", "Fe", "Wi", "Re",
    "Bu", "Br", "Sp", "Tr",
}


class Dictionary:
    """Persistent English→Loga code mapping. Grows during translation."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, str] = {}  # english_term → loga_code
        self.codes_used: set[str] = set(RESERVED_CODES)
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as f:
                data = json.load(f)
            self.entries = data.get("entries", {})
            self.codes_used = set(RESERVED_CODES)
            self.codes_used.update(self.entries.values())
            log.info("Dictionary loaded: %d entries", len(self.entries))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"entries": self.entries}, f, indent=2, ensure_ascii=False)

    def get(self, term: str) -> str | None:
        return self.entries.get(term)

    def add(self, term: str, code: str):
        if term not in self.entries:
            self.entries[term] = code
            self.codes_used.add(code)

    def relevant_entries(self, english_text: str) -> dict[str, str]:
        """Return only dictionary entries whose English term appears in the text."""
        text_lower = english_text.lower()
        matches = {}
        for term, code in self.entries.items():
            if term.startswith("_unknown_"):
                continue
            if term.lower() in text_lower:
                matches[term] = code
        return matches

    def format_for_prompt(self, english_text: str) -> str:
        """Format only the relevant dictionary entries for this article."""
        relevant = self.relevant_entries(english_text)
        if not relevant:
            return ""
        lines = ["PROPER NOUN CODES (use these exact codes for these entities):"]
        for term, code in sorted(relevant.items()):
            lines.append(f"  {term} = {code}")
        lines.append("For any proper noun NOT listed above, invent a unique 2-char code "
                      "(first char uppercase). Do NOT reuse any code from this list.")
        return "\n".join(lines)

    def extract_new_codes(self, english_text: str, loga_text: str):
        """Extract proper noun codes from output and cross-reference with input."""
        # Find all uppercase 2-char codes used as proper nouns in the Loga output
        # Pattern: uppercase letter + alphanumeric + case suffix (from !-/ range)
        proper_noun_pattern = re.compile(r'\b([A-Z][a-zA-Z0-9])[!"#$%&\'()*+,\-./]')
        loga_codes = set(proper_noun_pattern.findall(loga_text))

        # Remove codes that are known verbs (could appear with case suffixes as proper nouns)
        new_codes = loga_codes - self.codes_used

        if not new_codes:
            return

        # Try to match new codes to capitalized words in the English text
        english_words = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', english_text))

        for code in new_codes:
            # Find the English word that best matches this code
            best_match = None
            for word in english_words:
                # Check if the code could be an abbreviation of this word
                if word[0].upper() == code[0]:
                    if best_match is None or len(word) < len(best_match):
                        best_match = word
            if best_match:
                self.add(best_match, code)
            else:
                # Store with code as both key and value so we track it's used
                self.add(f"_unknown_{code}", code)
            self.codes_used.add(code)


# Seed dictionary with high-frequency Wikipedia proper nouns
SEED_DICTIONARY = {
    # Months
    "January": "Ja", "February": "Fb", "March": "Mc", "April": "Ap",
    "May": "My", "June": "Jn", "July": "Jy", "August": "Au",
    "September": "Sp", "October": "Oc", "November": "Nv", "December": "Dc",
    # Days
    "Monday": "Md", "Tuesday": "Td", "Wednesday": "Wd", "Thursday": "Tr",
    "Friday": "Fd", "Saturday": "Sd", "Sunday": "Sn",
    # Continents
    "Africa": "Af", "Asia": "As", "Europe": "Eu", "North America": "NA",
    "South America": "SA", "Antarctica": "An", "Australia": "AU", "Oceania": "Oa",
    # Major countries
    "United States": "US", "United Kingdom": "UK", "France": "Fr", "Germany": "De",
    "China": "CN", "Japan": "JP", "India": "IN", "Russia": "RU",
    "Brazil": "BR", "Canada": "CA", "Italy": "IT", "Spain": "ES",
    "Mexico": "MX", "England": "Eg", "Scotland": "Sc",
    "Ireland": "IE", "Netherlands": "NL", "Poland": "PL", "Sweden": "SE",
    "Norway": "NO", "Greece": "GR", "Egypt": "EG", "Turkey": "TK",
    "Israel": "IL", "Iran": "IR", "South Korea": "SK", "North Korea": "NK",
    "Argentina": "AR", "South Africa": "ZA", "New Zealand": "NZ",
    "Portugal": "PT", "Austria": "AT", "Switzerland": "CH", "Belgium": "BE",
    "Denmark": "DK", "Finland": "FI", "Hungary": "HU", "Romania": "RO",
    "Ukraine": "UA", "Pakistan": "PK", "Indonesia": "ID",
    "Philippines": "PH", "Vietnam": "VN", "Thailand": "TH",
    "Colombia": "CO", "Chile": "CL", "Peru": "PE", "Cuba": "CU",
    # Major cities
    "London": "Ld", "Paris": "Ps", "New York": "NY", "Tokyo": "Ty",
    "Beijing": "Bj", "Moscow": "Mw", "Berlin": "Bn", "Rome": "Rm",
    "Madrid": "Ma", "Washington": "DC", "Los Angeles": "LA",
    "Chicago": "Cg", "Sydney": "Sy", "Mumbai": "Mb",
    # History
    "World War": "WW", "United Nations": "UN", "European Union": "EU",
    # Misc
    "Earth": "Er", "Sun": "So", "Moon": "Mn", "God": "Gd",
    "Bible": "Bb", "Olympic": "OL", "Wikipedia": "WP",
    "Christian": "Cr", "Islam": "Im", "Catholic": "Ct", "Protestant": "Pr",
    "Latin": "Lt", "Greek": "Gk", "English": "En", "French": "Fh",
    "German": "Gn", "Spanish": "Sh", "Chinese": "Cz", "Japanese": "Jz",
    "Arabic": "Ab", "Russian": "Rs", "Hindi": "Hd",
    "Atlantic": "Ao", "Pacific": "Pf", "Indian Ocean": "IO",
    "Mediterranean": "Me", "Amazon": "Az", "Nile": "Ni",
    "Sahara": "Sr", "Himalaya": "Hm", "Alps": "Al",
}


def init_dictionary(path: Path) -> Dictionary:
    """Load or create dictionary, seeding with known proper nouns."""
    d = Dictionary(path)
    if not d.entries:
        for term, code in SEED_DICTIONARY.items():
            d.add(term, code)
        d.save()
        log.info("Dictionary initialized with %d seed entries", len(SEED_DICTIONARY))
    return d


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None


def get_model(model_name: str):
    global _model, _tokenizer
    if _model is None:
        from mlx_lm import load
        log.info("Loading model %s ...", model_name)
        _model, _tokenizer = load(model_name)
        log.info("Model loaded.")
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_text(
    text: str, model_name: str, dictionary: Dictionary, max_tokens: int = 4096,
) -> str:
    """Translate English text to Loga using local LLM."""
    from mlx_lm import generate

    model, tokenizer = get_model(model_name)

    chunks = _chunk_text(text, max_chars=3000)
    translated = []

    for chunk in chunks:
        dict_section = dictionary.format_for_prompt(chunk)
        system = GRAMMAR_PREAMBLE
        if dict_section:
            system = system + "\n\n" + dict_section

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Translate this English text to Loga:\n\n{chunk}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=max_tokens, verbose=True,
        )
        translated.append(response.strip())

    return "\n\n".join(translated)


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text on paragraph boundaries, keeping chunks under max_chars."""
    paragraphs = text.split("\n\n")
    chunks, current = [], []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    translated: int = 0
    skipped: int = 0
    failed: int = 0
    total_input_chars: int = 0
    total_output_chars: int = 0
    start_time: float = 0.0

    def rate(self) -> float:
        elapsed = time.time() - self.start_time
        return self.translated / elapsed * 3600 if elapsed > 0 else 0

    def eta(self, remaining: int) -> str:
        rate = self.rate()
        if rate <= 0:
            return "unknown"
        hours = remaining / rate
        if hours < 1:
            return f"{hours * 60:.0f}m"
        return f"{hours:.1f}h"


def run_pipeline(
    input_path: Path,
    output_path: Path,
    model_name: str,
    max_articles: int,
    max_tokens: int,
    dict_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary = init_dictionary(dict_path)

    # Load already-translated IDs
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        log.info("Resuming: %d articles already translated", len(done_ids))

    # Load input articles
    articles = []
    with open(input_path) as f:
        for line in f:
            try:
                d = json.loads(line)
                if d["id"] not in done_ids:
                    articles.append(d)
            except (json.JSONDecodeError, KeyError):
                continue

    if max_articles > 0:
        articles = articles[:max_articles]

    log.info("Articles to translate: %d", len(articles))
    if not articles:
        log.info("Nothing to do.")
        return

    # Pre-load model before starting the timer
    get_model(model_name)

    stats = Stats(start_time=time.time())

    with open(output_path, "a") as out_f:
        for article in tqdm(articles, desc="Translating"):
            try:
                text_loga = translate_text(
                    article["text"], model_name, dictionary, max_tokens,
                )

                # Extract new codes and grow the dictionary
                dictionary.extract_new_codes(article["text"], text_loga)

                record = {
                    "id": article["id"],
                    "title": article["title"],
                    "text": text_loga,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                stats.translated += 1
                stats.total_input_chars += len(article["text"])
                stats.total_output_chars += len(text_loga)

                # Save dictionary every 10 articles
                if stats.translated % 10 == 0:
                    dictionary.save()
                    remaining = len(articles) - stats.translated - stats.skipped - stats.failed
                    log.info(
                        "Progress: %d done, %.0f articles/hr, ETA %s, dict size %d",
                        stats.translated, stats.rate(), stats.eta(remaining),
                        len(dictionary.entries),
                    )

            except Exception as e:
                stats.failed += 1
                log.warning("Failed on article %s: %s", article.get("id", "?"), e)
                continue

    dictionary.save()
    elapsed = time.time() - stats.start_time
    log.info(
        "Done. %d translated, %d failed, %.1f min elapsed, "
        "%.0f articles/hr, dict size %d, "
        "input %d chars → output %d chars (%.1fx compression)",
        stats.translated, stats.failed, elapsed / 60,
        stats.rate(), len(dictionary.entries),
        stats.total_input_chars, stats.total_output_chars,
        stats.total_input_chars / max(stats.total_output_chars, 1),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path),
              required=True, help="Path to simplewiki-articles.jsonl")
@click.option("--output", "output_path", type=click.Path(path_type=Path),
              default="data/loga-articles.jsonl", show_default=True)
@click.option("--model", "model_name",
              default="mlx-community/gemma-4-31b-it-8bit", show_default=True,
              help="MLX model to use for translation")
@click.option("--max-articles", default=0, show_default=True,
              help="Limit number of articles (0 = all)")
@click.option("--max-tokens", default=4096, show_default=True,
              help="Max tokens per translation chunk")
@click.option("--dictionary", "dict_path", type=click.Path(path_type=Path),
              default="data/dictionary.json", show_default=True,
              help="Path to persistent dictionary file")
def cli(input_path, output_path, model_name, max_articles, max_tokens, dict_path):
    """Translate Simple English Wikipedia articles to Loga using a local LLM."""
    run_pipeline(input_path, output_path, model_name, max_articles, max_tokens, dict_path)


if __name__ == "__main__":
    cli()
