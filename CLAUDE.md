# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**loga-corpus** translates Simple English Wikipedia into Loga (a constructed language optimized for LLM tokenization efficiency) using a local LLM (Gemma via mlx-lm on Apple Silicon).

The output corpus (`data/loga-articles.jsonl`) feeds into the sibling project `~/Dev/conlang-experiment/` for pre-training efficiency experiments comparing English vs Loga.

## Common Commands

```bash
pip install -e .

# Translate (resumable — safe to interrupt and restart)
python -m translator.translate \
    --input ../conlang-experiment/data/raw/english/simplewiki-articles.jsonl \
    --output data/loga-articles.jsonl \
    --model mlx-community/gemma-3-27b-it-4bit \
    --max-articles 100    # omit for full corpus
```

## Key Design Decisions

- **Local LLM** (not API) to eliminate cost. Gemma 27B 4-bit fits in ~17GB on M4 Max 64GB.
- **Resumable**: output is append-only JSONL; already-translated article IDs are skipped on restart.
- **Loga grammar preamble** is injected as system prompt for every translation call — identical to the spec in `~/Dev/loga/conlang-spec.md`.
- **Translation confound** is the main reviewer risk. Mitigations: back-translation validation, Esperanto baseline from conlang-experiment (natural corpus, no translation step).

## Relationship to Other Projects

- `~/Dev/loga/` — Loga language spec, conlang design, Paper 2 staging
- `~/Dev/conlang-experiment/` — experiment harness (autoresearch-mlx), English/Esperanto/Loga pre-training runs
- This project produces the Loga corpus that conlang-experiment consumes
