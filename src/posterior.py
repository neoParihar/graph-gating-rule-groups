"""The posterior over remembered layouts (gating.ipynb, "The posterior over layouts").

    P(L | e) ∝ P(L) · P(k̂ | L) · P(x | L)^(1/T) · p(c | x, L)

over every vendor PageLayout with the page's number, plus NEW ("a vendor or layout not in memory"), which behaves
like the ALL_VENDORS tree and has no rules of its own.
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
from src.memory_graph import ALL_VENDORS, BOX_KEYS, FEATURES, MemoryGraph

TERMS = ("prior", "name", "fields", "boxes")
TERM_SETS = {"prior only": ("prior",), "name": ("prior", "name"), "fields": ("prior", "fields"),
             "boxes": ("prior", "boxes"), "fields + boxes": ("prior", "fields", "boxes"), "all": TERMS}


def presence(page: dict) -> np.ndarray:
    """x: 1 for every filled entity and every table present on the page, in FEATURES order."""
    filled = {k for k, f in page["fields"].items() if f["value"]} | {t["type"] for t in page["tables"] if t["rows"]}
    return np.array([float(f in filled) for f in FEATURES])


class LayoutPosterior:
    def __init__(self, graph: MemoryGraph, corpus: Corpus, cfg: Config):
        self.cfg = cfg
        layouts = graph.layouts()
        self.vendor_layouts = {i: L for i, L in layouts.items() if L["vendor"] != ALL_VENDORS}
        self.corpus_layouts = {L["page"]: L for L in layouts.values() if L["vendor"] == ALL_VENDORS}
        self.theta_all = {page: L["n_filled"] / L["docs"] for page, L in self.corpus_layouts.items()}
        self.known_keys = {L["vendor"] for L in self.vendor_layouts.values()}
        train_count = corpus.train_count
        self.p_new = sum(1 for n in train_count.values() if n == 1) / len(corpus.train_ids)   # Good-Turing N1/N

    def with_config(self, cfg: Config) -> "LayoutPosterior":
        """The same posterior with other settings (for sweeps); the graph is not read again."""
        other = object.__new__(LayoutPosterior)
        other.__dict__.update(self.__dict__)
        other.cfg = cfg
        return other

    def with_settings(self, **changes) -> "LayoutPosterior":
        return self.with_config(replace(self.cfg, **changes))

    def theta(self, L: dict | str, page: int, m: float | None = None) -> np.ndarray:
        """Smoothed P(feature present | L): the layout's counts shrunk toward the corpus with m pseudo-pages."""
        m = self.cfg.field_smoothing_m if m is None else m
        if L == NEW:
            return self.theta_all[page]
        return (L["n_filled"] + m * self.theta_all[page]) / (L["docs"] + m)

    def scores(self, page: dict, vendor_name: str, m: float | None = None) -> pd.DataFrame:
        """One row per candidate layout (and NEW): the log of each factor of Bayes' rule, before normalization."""
        cfg = self.cfg
        m = cfg.field_smoothing_m if m is None else m
        pn = min(page["number"], 2)
        candidates = [L for L in self.vendor_layouts.values() if L["page"] == pn]
        pages_here = sum(L["docs"] for L in candidates)
        x = presence(page)
        k_hat = H.vendor_key(vendor_name)
        n_keys = len(self.known_keys)
        centres = {k: H.bbox_center(f["bbox"]) for k, f in page["fields"].items() if f["bbox"] and k in BOX_KEYS}
        rows = []
        for L in candidates + [NEW]:
            is_new = L == NEW
            # 1. prior: Good-Turing mass for NEW, the rest by training pages per layout
            prior = self.p_new if is_new else (1 - self.p_new) * L["docs"] / pages_here
            # 2. name: noisy channel over vendor keys (exact, variant spelling, unrelated); no name is no evidence
            if k_hat == H.UNCLASSIFIED:
                name = 0.0
            elif is_new:
                name = math.log(1 - cfg.name_error) if k_hat not in self.known_keys else \
                    math.log(cfg.name_error / n_keys)
            elif k_hat == L["vendor"]:
                name = math.log(1 - cfg.name_error)
            elif H.keys_related(k_hat, L["vendor"]):
                name = math.log(cfg.name_error * cfg.name_variant)
            else:
                name = math.log(cfg.name_error * (1 - cfg.name_variant) / n_keys)
            # 3. fields: Bernoulli naive Bayes through the extraction channel, tempered by 1/T
            th = self.theta(L, pn, m)
            p1 = th * (1 - cfg.field_miss) + (1 - th) * cfg.field_false_fill
            fields = float(np.sum(x * np.log(p1) + (1 - x) * np.log(1 - p1))) / cfg.field_temperature
            # 4. boxes: piecewise-uniform density around each of L's ranges; 1 (log 0) where L has no range
            boxes = 0.0
            for k, c in centres.items():
                rule = None if is_new else L["rules"].get(k)
                if rule:
                    area = max(H.range_area(rule["range"]), 1e-6)
                    boxes += math.log(rule["q"] / area) if H.in_range(c, rule["range"]) else \
                        math.log((1 - rule["q"]) / (1 - area))
            rows.append({"layout_id": NEW if is_new else L["id"], "vendor": NEW if is_new else L["vendor"],
                         "prior": math.log(prior), "name": name, "fields": fields, "boxes": boxes})
        return pd.DataFrame(rows).set_index("layout_id")

    @staticmethod
    def posterior(scores: pd.DataFrame, terms: tuple[str, ...] = TERMS) -> dict[str, float]:
        """P(L | e) over the candidates, using only the chosen factors (for ablation)."""
        return H.normalize_log(scores[list(terms)].sum(axis=1).to_dict())

    @staticmethod
    def vendor_posterior(scores: pd.DataFrame, post: dict[str, float]) -> dict[str, float]:
        """P(vendor | e): the posterior summed over each vendor's layouts."""
        out = defaultdict(float)
        for layout_id, p in post.items():
            out[scores.at[layout_id, "vendor"]] += p
        return dict(out)

    def explain(self, page: dict, vendor_name: str, top: int = 6) -> pd.DataFrame:
        """The top candidates with their log-terms, total and posterior: the table behind the term chart."""
        s = self.scores(page, vendor_name)
        s["total"] = s[list(TERMS)].sum(axis=1)
        s["posterior"] = pd.Series(self.posterior(s))
        return s.sort_values("posterior", ascending=False).head(top)
