"""The Gemini calls: call 0 reads a PDF unguided, and its output is the evidence for the gate; the gated pass reads
the PDF again with the rule-group guidance appended to the system prompt.

    passes = GeminiPasses(cfg, corpus, gate)
    passes.run(sorted(corpus.test_ids))          # both calls per document, outputs saved under results/

Saved outputs are reused: call 0 is never repeated for a document, and the gated pass is repeated only when the
guidance text for it has changed. Every output is saved in the ground-truth output schema, so it scores the same
way (src/evaluation.py). Prompts are in src/prompts/.
"""

from __future__ import annotations

import json
from pathlib import Path

from google import genai

import helpers as H
from src.config import Config
from src.data import Corpus
from src.gate import Gate
from src.resources import load_prompt


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


class GeminiPasses:
    def __init__(self, cfg: Config, corpus: Corpus, gate: Gate):
        self.cfg, self.corpus, self.gate = cfg, corpus, gate
        self.client = genai.Client(vertexai=True, project=cfg.gcp_project, location=cfg.gcp_location)
        self.system_prompt = load_prompt("extract").format(catalog=H.entity_catalog_text())
        self.schema = H.gemini_response_schema()
        self.results = Path(cfg.results_dir)

    def _extract(self, doc_id: str, system_prompt: str) -> list[dict]:
        """One Gemini call on the document's PDF, returned in the output schema."""
        prediction = H.gemini_extract(self.client, self.cfg.ocr_model, self.corpus.pdf(doc_id), system_prompt,
                                      self.schema)
        return H.prediction_to_output_schema(prediction, self.corpus.page_facts[doc_id], self.cfg.ocr_model)

    def unguided(self, doc_id: str) -> dict:
        """Call 0: the document read with no memory, as a flat record (the evidence the gate sees)."""
        path = self.results / "unguided" / f"{doc_id}.json"
        if not path.exists():
            save_json(path, self._extract(doc_id, self.system_prompt))
        return H.document_to_record(doc_id, json.loads(path.read_text()), self.corpus.page_facts[doc_id])

    def gated_guidance(self, evidence: dict) -> tuple[list[dict], str]:
        """The gate's guide for every page of call 0's record, and the guidance text the model is given."""
        guides = [self.gate.page_guide(page, evidence["vendor_name"]) for page in evidence["pages"]]
        return guides, "\n\n".join(self.gate.guide_text(g) for g in guides)

    def gated(self, doc_id: str, evidence: dict | None = None) -> Path:
        """The gated pass: the guidance from call 0's evidence appended to the system prompt. The guide and the
        output are saved under results/guides/<gated_pass_name>/ and results/<gated_pass_name>/."""
        evidence = evidence or self.unguided(doc_id)
        guides, text = self.gated_guidance(evidence)
        name = self.cfg.gated_pass_name
        guide_path, out = self.results / "guides" / name / f"{doc_id}.json", self.results / name / f"{doc_id}.json"
        stale = guide_path.exists() and json.loads(guide_path.read_text())["guidance_text"] != text
        save_json(guide_path, {"guides": guides, "guidance_text": text})
        if stale or not out.exists():
            prompt = self.system_prompt + load_prompt("gated_suffix").format(rules=text)
            save_json(out, self._extract(doc_id, prompt))
        return out

    def run(self, doc_ids: list[str]) -> None:
        """Call 0 and the gated pass for each document, in order."""
        for i, doc_id in enumerate(doc_ids, 1):
            self.gated(doc_id, self.unguided(doc_id))
            print(f"\r{i}/{len(doc_ids)} documents", end="")
        print()
