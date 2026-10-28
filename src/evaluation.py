"""Scoring the gate on generated test pages (no LLM), and the short LLM-accuracy summary from saved Gemini passes.

A rule's **hit** is objective: the ground-truth value centre lies inside the range of the rule the gate considered.
The confidence is a statement about *where*, not *whether*, so only entities printed with a usable box count.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import helpers as H
from src.data import BUCKET_ORDER, NEW, Corpus
from src.gate import Gate
from src.posterior import TERM_SETS, RuleGroupPosterior
from src.scenarios import Case


# --- Gate on generated pages ------------------------------------------------------------------------------------


def evaluate(gate: Gate, cases: list[Case]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per generated page, one row per considered rule with a confidence)."""
    pages, rules = [], []
    for c in cases:
        g = gate.page_guide(c.evidence, c.vendor_name)
        decisions = [r["gate"] for r in g["rules"] if r["confidence"] is not None]
        base = {"case_id": c.case_id, "study": c.study, "scenario": c.scenario, "doc_id": c.doc_id, "page": c.page,
                "truth": c.truth, "true_vendor": c.true_vendor}
        pages.append({**base, "map_vendor": g["chosen_vendor"], "correct": g["chosen_vendor"] == c.truth,
                      "p_true": g["vendor_probabilities"].get(c.truth, 0.0), "p_new": g["p_new"],
                      "p_map": g["p_chosen"], "map_is_new": g["chosen"] == NEW,
                      **{f"n_{d}": decisions.count(d) for d in ("box", "label", "withheld")}})
        agree = c.meta.get("agree", {})
        for r in g["rules"]:
            f = c.gt["fields"].get(r["name"])
            if r["confidence"] is None or not f or not f["usable"]:
                continue
            rules.append({**base, "entity": r["name"], "confidence": r["confidence"], "gate": r["gate"],
                          "strength": r["strength"], "reliability": r["reliability"], "map_vendor": g["chosen_vendor"],
                          "map_correct": g["chosen_vendor"] == c.truth,
                          "twin_agrees": agree.get(r["name"]) if agree else None,
                          "hit": H.point_in_range(H.bbox_center(f["bbox"]), r["range"])})
    return pd.DataFrame(pages), pd.DataFrame(rules)


def shown_positions(pages: pd.DataFrame, rules: pd.DataFrame, by=("study", "scenario")) -> pd.DataFrame:
    """Per scenario, the positions the gate puts in front of the model: boxes per page, how many are right
    (precision), and wrong boxes per page. Wrong positions are what misleads an LLM."""
    by = list(by)
    n_pages = pages.groupby(by, sort=False).size()
    hit = rules[rules.gate == "box"].groupby(by, sort=False).hit
    out = pd.DataFrame({"pages": n_pages, "boxes_per_page": hit.size() / n_pages, "precision": hit.mean(),
                        "wrong_per_page": (hit.size() - hit.sum()) / n_pages}).reindex(n_pages.index)
    return out.fillna({"boxes_per_page": 0.0, "wrong_per_page": 0.0}).round(3)


def reliability(prob: pd.Series, outcome: pd.Series, bins=(0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0001)) -> pd.DataFrame:
    """Binned predicted probability against the observed frequency: calibrated means the two columns agree."""
    df = pd.DataFrame({"p": prob.astype(float), "y": outcome.astype(float)})
    df["bin"] = pd.cut(df.p, bins, right=False)
    return df.groupby("bin", observed=True).agg(n=("y", "size"), mean_predicted=("p", "mean"),
                                                observed=("y", "mean")).round(3)


def calibration_scores(rules: pd.DataFrame) -> dict[str, float]:
    """Brier score (0.25 is what always saying 0.5 earns) and ECE (count-weighted reliability gap)."""
    if rules.empty:
        return {"rules": 0, "brier": np.nan, "ece": np.nan}
    table = reliability(rules.confidence, rules.hit)
    return {"rules": len(rules), "brier": float(np.mean((rules.confidence - rules.hit) ** 2)),
            "ece": float((table.n * (table.mean_predicted - table.observed).abs()).sum() / table.n.sum())}


def coverage_table(rules: pd.DataFrame, taus=np.arange(0.1, 1.0, 0.1).round(1)) -> pd.DataFrame:
    """Risk–coverage: for each threshold, the share of rules shown and how often shown/withheld rules were right."""
    return pd.DataFrame([{"tau": t, "coverage": (rules.confidence >= t).mean(),
                          "accuracy_of_shown": rules[rules.confidence >= t].hit.mean(),
                          "accuracy_of_withheld": rules[rules.confidence < t].hit.mean()} for t in taus])


def study_summary(pages: pd.DataFrame, rules: pd.DataFrame, by=("study", "scenario")) -> pd.DataFrame:
    """Per scenario: did the posterior find the right group, what did the gate do, and was it right to?

    * map_acc: the MAP vendor is the right answer (the true vendor, or NEW for an unseen vendor / layout)
    * p_true, p_new: mean probability of the right answer, and of NEW
    * share_box / share_label / share_withheld: gate decisions over the rules considered
    * hit_box / hit_withheld: how often the value really was in the range, for shown vs withheld rules
    """
    by = list(by)
    p = pages.groupby(by, sort=False).agg(pages=("page", "size"), map_acc=("correct", "mean"),
                                          p_true=("p_true", "mean"), p_new=("p_new", "mean"))
    r = rules.groupby(by, sort=False).agg(
        rules=("hit", "size"), mean_conf=("confidence", "mean"),
        share_box=("gate", lambda g: (g == "box").mean()), share_label=("gate", lambda g: (g == "label").mean()),
        share_withheld=("gate", lambda g: (g == "withheld").mean()),
        hit_box=("hit", lambda h: h[rules.loc[h.index, "gate"] == "box"].mean()),
        hit_withheld=("hit", lambda h: h[rules.loc[h.index, "gate"] == "withheld"].mean()),
        brier=("hit", lambda h: float(np.mean((rules.loc[h.index, "confidence"] - h) ** 2))))
    return p.join(r).round(3)


def identification(posterior: RuleGroupPosterior, cases: list[Case]) -> pd.DataFrame:
    """Which evidence identifies the group? Per scenario and evidence set: top-1 accuracy of the MAP vendor, mean
    probability of the right answer, and log loss (punishes confident mistakes)."""
    rows = []
    for c in cases:
        log_terms = posterior.evidence_log_terms(c.evidence, c.vendor_name)
        for set_name, terms in TERM_SETS.items():
            vp = posterior.vendor_probabilities(log_terms, posterior.group_probabilities(log_terms, terms))
            best = max(vp, key=vp.get)
            rows.append({"study": c.study, "scenario": c.scenario, "evidence": set_name, "correct": best == c.truth,
                         "p_true": vp.get(c.truth, 0.0)})
    df = pd.DataFrame(rows)
    return df.groupby(["study", "scenario", "evidence"], sort=False).agg(
        pages=("correct", "size"), top1_acc=("correct", "mean"), p_true=("p_true", "mean"),
        log_loss=("p_true", lambda p: float(np.mean(-np.log(np.clip(p, 1e-12, 1)))))).round(3)


def twin_agreement(rules: pd.DataFrame) -> pd.DataFrame:
    """E2: rules where the twins' ranges overlap against rules where they don't, per scenario."""
    df = rules[rules.study == "E2 twins"].dropna(subset=["twin_agrees"]).copy()
    df["twins"] = df.twin_agrees.map({True: "ranges overlap", False: "ranges differ"})
    return df.groupby(["scenario", "twins"], sort=False).agg(
        rules=("hit", "size"), mean_conf=("confidence", "mean"), hit_rate=("hit", "mean"),
        share_box=("gate", lambda g: (g == "box").mean()),
        share_withheld=("gate", lambda g: (g == "withheld").mean())).round(3)


# --- Part IV: saved Gemini passes (gating.ipynb's evaluation, shortened) ----------------------------------------

LLM_SETUPS = {"unguided": "unguided", "gated memory": "pass2"}
LLM_METRICS = ["value_exact_acc", "false_fill", "mean_bbox_overlap", "center_within_tol"]


def score_prediction(pred: dict, gt: dict, tolerance: float) -> list[dict]:
    """One row per (page, entity) of one prediction against ground truth."""
    rows = []
    pred_pages = {p["number"]: p for p in pred["pages"]}
    for gt_page in gt["pages"]:
        pp = pred_pages.get(gt_page["number"], {"fields": {}})
        for key, g in gt_page["fields"].items():
            if key in H.DEFAULTED_KEYS or not g["valid"]:
                continue
            p = pp["fields"].get(key, {"value": "", "bbox": None})
            rows.append({
                "doc_id": gt["doc_id"], "page": gt_page["number"], "entity": key,
                "gt_filled": bool(g["value"]), "pred_filled": bool(p["value"]),
                "exact": H.normalize_value_for_match(p["value"]) == H.normalize_value_for_match(g["value"]),
                "bbox_overlap": H.bbox_iou(p["bbox"], g["bbox"]) if g["usable"] else None,
                "center_within_tol": (bool(p["bbox"]) and H.point_in_range(H.bbox_center(p["bbox"]), H.square_range(
                    *H.bbox_center(g["bbox"]), tolerance))) if g["usable"] else None,
            })
    return rows


def summarize(df: pd.DataFrame) -> pd.Series:
    filled, boxed = df[df.gt_filled], df.dropna(subset=["bbox_overlap"])
    return pd.Series({"value_exact_acc": filled.exact.mean(),
                      "false_fill": (~df.gt_filled & df.pred_filled).sum() / max((~df.gt_filled).sum(), 1),
                      "mean_bbox_overlap": boxed.bbox_overlap.mean(),
                      "center_within_tol": boxed.center_within_tol.astype(float).mean()})


def llm_fields(corpus: Corpus) -> pd.DataFrame | None:
    """Every saved pass output scored against ground truth, or None when results/ holds no passes."""
    results = corpus.cfg.results_dir
    rows = []
    for setup, folder in LLM_SETUPS.items():
        for doc_id in sorted(corpus.test_ids):
            path = Path(results) / folder / f"{doc_id}.json"
            if not path.exists():
                continue
            pred = H.document_to_record(doc_id, json.loads(path.read_text()), corpus.page_facts[doc_id])
            rows += [{**r, "setup": setup}
                     for r in score_prediction(pred, corpus.records[doc_id], corpus.cfg.bbox_tolerance)]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["bucket"] = df.doc_id.map(lambda d: corpus.bucket(corpus.vendor_of(d)))
    return df


def llm_table(fields: pd.DataFrame) -> pd.DataFrame:
    """Metrics per training-set size bucket and setup, setups side by side."""
    out = fields.groupby(["bucket", "setup"]).apply(summarize, include_groups=False)[LLM_METRICS].unstack("setup")
    out = out.reindex([b for b in BUCKET_ORDER if b in out.index])
    return out.reindex(columns=[(m, s) for m in LLM_METRICS for s in LLM_SETUPS if (m, s) in out.columns]).round(3)
