"""Which rule group is this page? The posterior over the remembered rule groups.

    P(G | e) ∝ P(G) · P(k̂ | G) · P(x | G)^(1/T) · p(c | x, G)

over every rule group of the page's page type, plus NEW ("a page unlike any group in memory"), which behaves like
the corpus-wide view and has no rules of its own. Each factor is the soft version of one check of the ingestion
rule: the vendor name, the nodes present, and whether the boxes lie in the rules' ranges.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pandas as pd

import helpers as H
from src.config import Config
from src.data import NEW, Corpus
from src.rule_memory import RuleMemory
from src.schema import BOX_KEYS, CORPUS_VIEW, page_presence, page_type

TERMS = ("prior", "name", "fields", "boxes")
TERM_SETS = {"prior only": ("prior",), "name": ("prior", "name"), "fields": ("prior", "fields"),
             "boxes": ("prior", "boxes"), "fields + boxes": ("prior", "fields", "boxes"), "all": TERMS}


class RuleGroupPosterior:
    def __init__(self, memory: RuleMemory, corpus: Corpus, cfg: Config, p_new: float | None = None):
        """`p_new` is the prior mass of NEW; by default the memory's Good–Turing estimate (`memory.p_new`)."""
        self.cfg = cfg
        groups = memory.rule_groups()
        self.groups = {i: G for i, G in groups.items() if G["vendor"] != CORPUS_VIEW}
        self.corpus_views = {G["page"]: G for G in groups.values() if G["vendor"] == CORPUS_VIEW}
        self.corpus_fill_rates = {page: V["filled_pages"] / V["pages"] for page, V in self.corpus_views.items()}
        self.known_vendors = {G["vendor"] for G in self.groups.values()}
        self.p_new = memory.p_new if p_new is None else p_new

    def with_config(self, cfg: Config) -> "RuleGroupPosterior":
        """The same posterior with other settings (for sweeps); the memory is not read again."""
        other = object.__new__(RuleGroupPosterior)
        other.__dict__.update(self.__dict__)
        other.cfg = cfg
        return other

    def with_settings(self, **changes) -> "RuleGroupPosterior":
        return self.with_config(replace(self.cfg, **changes))

    def smoothed_fill_rates(self, group: dict | str, pt: int, m: float | None = None) -> np.ndarray:
        """P(feature present | G): the group's fill rates shrunk toward the corpus view with m pseudo-pages. NEW
        uses the corpus view itself."""
        m = self.cfg.field_smoothing_m if m is None else m
        if group == NEW:
            return self.corpus_fill_rates[pt]
        return (group["filled_pages"] + m * self.corpus_fill_rates[pt]) / (group["pages"] + m)

    def evidence_log_terms(self, page: dict, vendor_name: str, m: float | None = None) -> pd.DataFrame:
        """One row per candidate group (and NEW): the log of each factor of Bayes' rule, before normalization."""
        cfg = self.cfg
        m = cfg.field_smoothing_m if m is None else m
        pt = page_type(page)
        candidates = [G for G in self.groups.values() if G["page"] == pt]
        pages_here = sum(G["pages"] for G in candidates)
        present = np.array(page_presence(page))
        read_key = H.vendor_key(vendor_name)
        n_vendors = len(self.known_vendors)
        centres = {k: H.bbox_center(f["bbox"]) for k, f in page["fields"].items() if f["bbox"] and k in BOX_KEYS}
        rows = []
        for G in candidates + [NEW]:
            is_new = G == NEW
            # 1. prior: Good–Turing mass for NEW, the rest by training pages per group
            prior = self.p_new if is_new else (1 - self.p_new) * G["pages"] / pages_here
            # 2. name: noisy channel over vendor keys (exact, variant spelling, unrelated); no name is no evidence
            if read_key == H.UNCLASSIFIED:
                name = 0.0
            elif is_new:
                name = math.log(1 - cfg.name_error) if read_key not in self.known_vendors else \
                    math.log(cfg.name_error / n_vendors)
            elif read_key == G["vendor"]:
                name = math.log(1 - cfg.name_error)
            elif H.vendor_keys_related(read_key, G["vendor"]):
                name = math.log(cfg.name_error * cfg.name_variant)
            else:
                name = math.log(cfg.name_error * (1 - cfg.name_variant) / n_vendors)
            # 3. fields: Bernoulli naive Bayes through the extraction channel, tempered by 1/T
            rates = self.smoothed_fill_rates(G, pt, m)
            p_seen = rates * (1 - cfg.field_miss) + (1 - rates) * cfg.field_false_fill
            fields = float(np.sum(present * np.log(p_seen) + (1 - present) * np.log(1 - p_seen))) \
                / cfg.field_temperature
            # 4. boxes: piecewise-uniform density around each of G's ranges; 1 (log 0) where G has no range
            boxes = 0.0
            for k, c in centres.items():
                rule = None if is_new else G["rules"].get(k)
                if rule:
                    area, r = max(H.range_area(rule["range"]), 1e-6), rule["reliability"]
                    boxes += math.log(r / area) if H.point_in_range(c, rule["range"]) else \
                        math.log((1 - r) / (1 - area))
            rows.append({"group": NEW if is_new else G["id"], "vendor": NEW if is_new else G["vendor"],
                         "prior": math.log(prior), "name": name, "fields": fields, "boxes": boxes})
        return pd.DataFrame(rows).set_index("group")

    @staticmethod
    def group_probabilities(log_terms: pd.DataFrame, terms: tuple[str, ...] = TERMS) -> dict[str, float]:
        """P(G | e) over the candidates, using only the chosen factors (for ablation)."""
        return H.normalize_log_probabilities(log_terms[list(terms)].sum(axis=1).to_dict())

    @staticmethod
    def vendor_probabilities(log_terms: pd.DataFrame, probabilities: dict[str, float]) -> dict[str, float]:
        """P(vendor | e): the group probabilities summed over each vendor's groups."""
        out = defaultdict(float)
        for group_id, p in probabilities.items():
            out[log_terms.at[group_id, "vendor"]] += p
        return dict(out)

    def explain(self, page: dict, vendor_name: str, top: int = 6, per_vendor: bool = False) -> pd.DataFrame:
        """The top candidates with their log-terms, total and probability: the table behind the term chart.
        per_vendor keeps one row per vendor, its most probable group, with the vendor's total probability
        (`vendor_probability`), how many of its groups competed (`groups_competing`) and the row group's training
        pages (`training_pages`)."""
        s = self.evidence_log_terms(page, vendor_name)
        s["total"] = s[list(TERMS)].sum(axis=1)
        s["probability"] = pd.Series(self.group_probabilities(s))
        s = s.sort_values("probability", ascending=False)
        if per_vendor:
            by_vendor = s.groupby("vendor")
            s = s.assign(vendor_probability=by_vendor.probability.transform("sum"),
                         groups_competing=by_vendor.probability.transform("size"),
                         training_pages=[0 if i == NEW else self.groups[i]["pages"] for i in s.index])
            s = s.drop_duplicates("vendor")
        return s.head(top)
