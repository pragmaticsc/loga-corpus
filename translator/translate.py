"""
translator/translate.py
=======================
English → Loga translation using a local LLM (Gemma via mlx-lm).

Reads Simple English Wikipedia articles from a JSONL file, translates each
article to Loga, and writes output as JSONL. Resumable — skips articles
already present in the output file.

Usage:
    python -m translator.translate \
        --input ../conlang-experiment/data/raw/english/simplewiki-articles.jsonl \
        --output data/loga-articles.jsonl \
        --model mlx-community/gemma-3-27b-it-4bit

Requirements:
    pip install -e .
    (downloads model weights on first run — ~17GB for 4-bit 27B)
"""

from __future__ import annotations

import json
import logging
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

Proper nouns: abbreviate to 2 printable ASCII chars, capitalize first.
  "England" → En!   "Paris" → Pa!   "Albert Einstein" → Ab! Es!
  "United States" → US!   "World War" → Ww!

SAMPLE SENTENCES:
  ka! da" Se: .     = The person sees the thing.
  mi! bo& Go; .     = I went toward the city.
  ku! li" Be: .     = Water is life.
  mi! ` ka! da" Se; " Kn: .  = I know that the person saw the thing.
  \\ ka! bo& Go< .   = Some people will go to the city.
  mi! da" _Se: .    = I do not see the thing.

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

def translate_text(text: str, model_name: str, max_tokens: int = 4096) -> str:
    """Translate English text to Loga using local LLM."""
    from mlx_lm import generate

    model, tokenizer = get_model(model_name)

    chunks = _chunk_text(text, max_chars=3000)
    translated = []

    for chunk in chunks:
        messages = [
            {"role": "system", "content": GRAMMAR_PREAMBLE},
            {"role": "user", "content": f"Translate this English text to Loga:\n\n{chunk}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=max_tokens, verbose=False,
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
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
                text_loga = translate_text(article["text"], model_name, max_tokens)
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

                if stats.translated % 10 == 0:
                    remaining = len(articles) - stats.translated - stats.skipped - stats.failed
                    log.info(
                        "Progress: %d done, %.0f articles/hr, ETA %s",
                        stats.translated, stats.rate(), stats.eta(remaining),
                    )

            except Exception as e:
                stats.failed += 1
                log.warning("Failed on article %s: %s", article.get("id", "?"), e)
                continue

    elapsed = time.time() - stats.start_time
    log.info(
        "Done. %d translated, %d failed, %.1f min elapsed, "
        "%.0f articles/hr, input %d chars → output %d chars (%.1fx compression)",
        stats.translated, stats.failed, elapsed / 60,
        stats.rate(),
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
              default="mlx-community/gemma-3-27b-it-4bit", show_default=True,
              help="MLX model to use for translation")
@click.option("--max-articles", default=0, show_default=True,
              help="Limit number of articles (0 = all)")
@click.option("--max-tokens", default=4096, show_default=True,
              help="Max tokens per translation chunk")
def cli(input_path, output_path, model_name, max_articles, max_tokens):
    """Translate Simple English Wikipedia articles to Loga using a local LLM."""
    run_pipeline(input_path, output_path, model_name, max_articles, max_tokens)


if __name__ == "__main__":
    cli()
