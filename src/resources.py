"""The text files next to the code: Cypher queries in src/queries/<name>.cypher, prompts in src/prompts/<name>.txt.
Each is read once and cached."""

from __future__ import annotations

from functools import cache
from pathlib import Path

QUERY_DIR = Path(__file__).parent / "queries"
PROMPT_DIR = Path(__file__).parent / "prompts"


@cache
def load_cypher(name: str) -> str:
    return (QUERY_DIR / f"{name}.cypher").read_text()


@cache
def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text()
