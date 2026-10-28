"""Generated test pages: controlled variants of held-out ground-truth pages, each with a known right answer.

They stand in for Gemini's unguided call (call 0), so the gate can be studied without an LLM. Every case carries
two pages:
* `evidence`: what the posterior sees (the GT page, transformed, then optionally passed through the noise channel)
* `gt`: what "the rule was right" is judged against (the GT page, or the transformed page when the transformation
  is meant to be real, as in a redesign)

Transformations
* noise channel: box centres jittered (mean miss `noise_box_miss`), printed fields dropped, unprinted fields filled
  at a random spot, the vendor name misread (variant spelling or typo)
* blank name: the vendor name is missing, so the name term carries no evidence
* rename: a known vendor's page under a name the memory has never seen
* swap name: the page of vendor A under the name of its structural twin B
* redesign: every box moved by `drift_shift`, under the vendor's own name
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import helpers as H
from src.config import Config
from src.data import NEW, Corpus
from src.rule_memory import RuleMemory
from src.schema import BOX_KEYS, page_type

_NAME_A = ["Harbor", "Ironwood", "Bluefield", "Copper Ridge", "Northgate", "Silverline", "Oakmont", "Redstone",
           "Lakeshore", "Summit", "Granite Peak", "Westbrook", "Pinecrest", "Brightwater", "Stonebridge"]
_NAME_B = ["Supply", "Logistics", "Medical", "Industrial", "Services", "Distribution", "Partners", "Systems",
           "Trading", "Works"]


@dataclass
class Case:
    """One generated test page."""
    case_id: str
    study: str
    scenario: str
    doc_id: str
    evidence: dict                  # page record the posterior sees
    vendor_name: str                # vendor name the posterior sees
    gt: dict                        # page record hits are judged against
    truth: str                      # the right answer for identification: a vendor key, or NEW
    true_vendor: str                # the document's real vendor key
    meta: dict = field(default_factory=dict)

    @property
    def page(self) -> int:
        return self.evidence["number"]


# --- Transformations --------------------------------------------------------------------------------------------


def move_box_centre(bbox: list[float], cx: float, cy: float) -> list[float]:
    """The box with its centre moved to (cx, cy), kept on the page."""
    x, y, w, h = bbox
    cx, cy = min(max(cx, w / 2), 1 - w / 2), min(max(cy, h / 2), 1 - h / 2)
    return [round(cx - w / 2, 4), round(cy - h / 2, 4), w, h]


def add_noise(page: dict, rng: random.Random, box_miss: float, field_miss: float, false_fill: float) -> dict:
    """The page as a noisy extraction would return it. Each axis of a box centre gets Gaussian noise with
    σ = box_miss / sqrt(π/2), so the mean distance from the true centre is box_miss."""
    page = copy.deepcopy(page)
    sigma = box_miss / math.sqrt(math.pi / 2)
    for key in BOX_KEYS:
        f = page["fields"].get(key)
        if f is None:
            continue
        if f["value"]:
            if rng.random() < field_miss:
                page["fields"][key] = {**f, "value": "", "bbox": None, "usable": False}
            elif f["bbox"] and sigma > 0:
                cx, cy = H.bbox_center(f["bbox"])
                f["bbox"] = move_box_centre(f["bbox"], cx + rng.gauss(0, sigma), cy + rng.gauss(0, sigma))
        elif rng.random() < false_fill:
            page["fields"][key] = {**f, "value": "?", "bbox": move_box_centre([0, 0, 0.1, 0.012], rng.random(),
                                                                                  rng.random()), "usable": False}
    return page


def shift_page(page: dict, dx: float, dy: float) -> dict:
    """A redesign: every value box moved by (dx, dy), kept on the page."""
    page = copy.deepcopy(page)
    for f in page["fields"].values():
        if f["bbox"]:
            cx, cy = H.bbox_center(f["bbox"])
            f["bbox"] = move_box_centre(f["bbox"], cx + dx, cy + dy)
    return page


def misread_name(name: str, rng: random.Random) -> str:
    """A variant spelling (an extra word) or a typo (two letters swapped in the longest word)."""
    if rng.random() < 0.5:
        return f"{name} {rng.choice(['Services', 'Group', 'USA', 'Holdings'])}"
    words = name.split()
    i = max(range(len(words)), key=lambda j: len(words[j]))
    w = words[i]
    if len(w) >= 4:
        j = rng.randrange(1, len(w) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    else:
        words[i] = w + "x"
    return " ".join(words)


def fresh_names(n: int, rng: random.Random, taken: set[str]) -> list[str]:
    """n vendor names whose keys are not in memory."""
    names = []
    while len(names) < n:
        name = f"{rng.choice(_NAME_A)} {rng.choice(_NAME_B)} {rng.randint(2, 99)}"
        if H.vendor_key(name) not in taken:
            taken.add(H.vendor_key(name))
            names.append(name)
    return names


# --- Structural twins -------------------------------------------------------------------------------------------


def find_twins(memory: RuleMemory, min_similarity: float, min_shared_rules: int = 1) -> pd.DataFrame:
    """Pairs of rule groups of *different* vendors, same page type, whose structure vectors (which nodes, and
    roughly where) have similarity >= min_similarity. Per pair: how many located rules they share, and on how many
    of those their ranges overlap (the twins "agree" on where the value is). Pairs sharing fewer than
    min_shared_rules located entities (e.g. near-empty continuation pages) are dropped."""
    rows = memory.similar_group_pairs(min_similarity)
    groups = memory.rule_groups()
    for r in rows:
        agree = rules_overlap(groups[r["group_a"]], groups[r["group_b"]])
        r["shared_rules"] = len(agree)
        r["agreeing_rules"] = sum(agree.values())
    df = pd.DataFrame(rows)
    return df[df.shared_rules >= min_shared_rules].reset_index(drop=True) if not df.empty else df


def rules_overlap(group_a: dict, group_b: dict) -> dict[str, bool]:
    """For every entity both groups locate: do their ranges overlap?"""
    return {k: H.range_overlap_area(r["range"], group_b["rules"][k]["range"]) > 0
            for k, r in group_a["rules"].items() if k in group_b["rules"]}


# --- The generator ----------------------------------------------------------------------------------------------


class ScenarioGenerator:
    """Builds the test-page studies from the held-out documents. Seeded: the same settings give the same cases."""

    def __init__(self, corpus: Corpus, memory: RuleMemory, cfg: Config):
        self.corpus, self.memory, self.cfg = corpus, memory, cfg
        self.test_ids = sorted(corpus.test_ids)
        self.known_names = memory.vendor_names()

    def _rng(self, *salt) -> random.Random:
        return random.Random(f"{self.cfg.scenario_seed}|" + "|".join(map(str, salt)))

    def _evidence(self, page: dict, name: str, noisy: bool, rng: random.Random,
                  box_miss: float | None = None) -> tuple[dict, str]:
        """The page and the name as the posterior will see them."""
        if not noisy:
            return page, name
        cfg = self.cfg
        page = add_noise(page, rng, cfg.noise_box_miss if box_miss is None else box_miss, cfg.noise_field_miss,
                         cfg.noise_false_fill)
        if name and rng.random() < cfg.noise_name_error:
            name = misread_name(name, rng)
        return page, name

    def _case(self, study: str, scenario: str, doc_id: str, page: dict, name: str, noisy: bool,
              truth: str | None = None, gt: dict | None = None, box_miss: float | None = None, **meta) -> Case:
        rng = self._rng(study, scenario, doc_id, page["number"], box_miss)
        evidence, seen_name = self._evidence(page, name, noisy, rng, box_miss)
        return Case(case_id=f"{study}|{scenario}|{doc_id}|p{page['number']}", study=study, scenario=scenario,
                    doc_id=doc_id, evidence=evidence, vendor_name=seen_name, gt=gt or page,
                    truth=truth or self.corpus.truth(doc_id), true_vendor=self.corpus.vendor_of(doc_id),
                    meta={"noisy": noisy, **meta})

    def _test_pages(self, seen_only: bool = False):
        for doc_id in self.test_ids:
            if seen_only and self.corpus.truth(doc_id) == NEW:
                continue
            rec = self.corpus.records[doc_id]
            for page in rec["pages"]:
                yield doc_id, rec, page

    # E1 -----------------------------------------------------------------------------------------------------------
    def baseline(self, box_miss: float | None = None) -> list[Case]:
        """Every held-out page, clean (the ground truth itself) and through the noise channel."""
        cases = []
        for doc_id, rec, page in self._test_pages():
            cases.append(self._case("E1 baseline", "clean", doc_id, page, rec["vendor_name"], noisy=False))
            cases.append(self._case("E1 baseline", "noisy", doc_id, page, rec["vendor_name"], noisy=True,
                                    box_miss=box_miss))
        return cases

    # E2 -----------------------------------------------------------------------------------------------------------
    def twins(self, twins: pd.DataFrame | None = None) -> list[Case]:
        """Test pages of a vendor with a structural twin, seen under its own name, no name, and the twin's name."""
        if twins is None:
            twins = find_twins(self.memory, self.cfg.twin_min_similarity, self.cfg.twin_min_shared_rules)
        groups = self.memory.rule_groups()
        partners: dict[tuple[str, int], list[dict]] = {}
        for r in twins.itertuples():
            for own, other, own_group, other_group in ((r.vendor_a, r.vendor_b, r.group_a, r.group_b),
                                                       (r.vendor_b, r.vendor_a, r.group_b, r.group_a)):
                partners.setdefault((own, r.page), []).append(
                    {"twin": other, "own_group": own_group, "twin_group": other_group, "sim": r.sim})
        cases = []
        for doc_id, rec, page in self._test_pages(seen_only=True):
            for pair in partners.get((rec["vendor_key"], page_type(page)), [])[:1]:   # the closest twin
                agree = rules_overlap(groups[pair["own_group"]], groups[pair["twin_group"]])
                twin_name = (self.known_names.get(pair["twin"]) or [pair["twin"]])[0]
                meta = {"twin": pair["twin"], "twin_similarity": pair["sim"], "agree": agree,
                        "own_group": pair["own_group"], "twin_group": pair["twin_group"]}
                for scenario, name in (("own name", rec["vendor_name"]), ("no name", ""), ("twin's name", twin_name)):
                    cases.append(self._case("E2 twins", scenario, doc_id, page, name, noisy=True, **meta))
        return cases

    # E3 -----------------------------------------------------------------------------------------------------------
    def no_vendor(self, box_miss: float | None = None) -> list[Case]:
        """Every held-out page with the vendor name missing: identification rests on structure alone."""
        cases = []
        for doc_id, rec, page in self._test_pages():
            cases.append(self._case("E3 no vendor", "no name, clean", doc_id, page, "", noisy=False))
            cases.append(self._case("E3 no vendor", "no name, noisy", doc_id, page, "", noisy=True,
                                    box_miss=box_miss))
        return cases

    # E4 -----------------------------------------------------------------------------------------------------------
    def unseen_and_drift(self, box_miss: float | None = None) -> list[Case]:
        """Real unseen vendors; known layouts under a name never seen; known names on a redesigned layout."""
        cases = []
        taken = set(self.known_names) | {self.corpus.vendor_of(d) for d in self.corpus.records}
        names = dict(zip(self.test_ids, fresh_names(len(self.test_ids), self._rng("fresh names"), taken)))
        dx, dy = self.cfg.drift_shift
        for doc_id, rec, page in self._test_pages():
            if self.corpus.truth(doc_id) == NEW:
                cases.append(self._case("E4 unseen & drift", "unseen vendor", doc_id, page, rec["vendor_name"],
                                        noisy=True, box_miss=box_miss))
                continue
            cases.append(self._case("E4 unseen & drift", "known layout, new name", doc_id, page, names[doc_id],
                                    noisy=True, box_miss=box_miss))
            moved = shift_page(page, dx, dy)
            cases.append(self._case("E4 unseen & drift", "known name, redesigned layout", doc_id, moved,
                                    rec["vendor_name"], noisy=True, truth=NEW, gt=moved, box_miss=box_miss))
        return cases

    def all_studies(self) -> list[Case]:
        return self.baseline() + self.twins() + self.no_vendor() + self.unseen_and_drift()


def case_table(cases: list[Case]) -> pd.DataFrame:
    """How many generated pages each study and scenario holds."""
    df = pd.DataFrame([{"study": c.study, "scenario": c.scenario, "doc_id": c.doc_id, "truth_is_new": c.truth == NEW}
                       for c in cases])
    return df.groupby(["study", "scenario"], sort=False).agg(
        pages=("doc_id", "size"), documents=("doc_id", "nunique"), truth_is_new=("truth_is_new", "mean")).round(2)


def describe_noise(cases: list[Case]) -> pd.Series:
    """What the noise channel actually did to the evidence, measured against the clean page."""
    dist, dropped, invented = [], 0, 0
    n_filled = n_empty = 0
    for c in cases:
        if not c.meta.get("noisy"):
            continue
        for key in BOX_KEYS:
            g, e = c.gt["fields"].get(key), c.evidence["fields"].get(key)
            if g is None or e is None:
                continue
            if g["value"]:
                n_filled += 1
                dropped += not e["value"]
                if g["bbox"] and e["bbox"]:
                    dist.append(H.bbox_center_distance(g["bbox"], e["bbox"]))
            else:
                n_empty += 1
                invented += bool(e["value"])
    return pd.Series({"mean_centre_miss": float(np.mean(dist)) if dist else 0.0,
                      "printed_fields_dropped": dropped / max(n_filled, 1),
                      "empty_fields_filled": invented / max(n_empty, 1)}).round(3)
