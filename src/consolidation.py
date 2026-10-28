"""Consolidation: from the observations supporting one rule node to its rule.

For one entity of one rule group:
1. peak: the observed box centre with the most other centres within `bbox_tolerance`; support is that count
2. range: a square around the peak, half-width clip(k · spread, box_range_min, box_range_max), where spread is the
   median distance of all centres from the peak
3. reliability: the chance the next page of the group prints the value inside the range, estimated leave-one-out
   (each centre against the range rebuilt from the others) with an empirical-Bayes Beta prior
4. strength: strong / weak when enough pages agree on the peak, single for a one-page group, otherwise none
"""

from __future__ import annotations

from collections import Counter

import numpy as np

import helpers as H
from src.config import Config


def density_peak(points: np.ndarray, tolerance: float) -> tuple[int, int]:
    """(index of the point with the most neighbours within `tolerance` on both axes, that neighbour count)."""
    near = (np.abs(points[:, None, :] - points[None, :, :]) <= tolerance).all(axis=2).sum(axis=1)
    i = int(np.argmax(near))
    return i, int(near[i])


def range_around_peak(points: np.ndarray, cfg: Config) -> tuple[list[float], np.ndarray, float]:
    """(range [x0, y0, x1, y1], peak, spread) of a set of box centres."""
    i, _ = density_peak(points, cfg.bbox_tolerance)
    peak = points[i]
    spread = float(np.median(np.hypot(*(points - peak).T)))
    half_width = min(max(cfg.box_range_spread_k * spread, cfg.box_range_min), cfg.box_range_max)
    return H.square_range(*peak, half_width), peak, spread


def leave_one_out_hits(points: np.ndarray, cfg: Config) -> int:
    """How many centres fall inside the range rebuilt from all the other centres. A page joins a group only when
    its centres are inside the group's ranges, so counting members inside the full range would flatter the rule."""
    return sum(H.point_in_range(tuple(points[i]), range_around_peak(np.delete(points, i, axis=0), cfg)[0])
               for i in range(len(points)))


def consolidate_rule(observations: list[dict], group_pages: int, cfg: Config, prior: tuple[float, float]) -> dict:
    """The rule of one node of a group from the observations supporting it (each {point, bbox, label}): the
    properties written onto the EntityRule or TableRule node. `prior` is the Beta(alpha, beta) for reliability."""
    alpha, beta = prior
    labels = Counter(o["label"] for o in observations if o["label"])
    rule = {"n_boxes": 0, "support": 0, "loo_hits": 0, "bbox_center": None, "bbox_median": None, "spread": None,
            "range": None, "reliability": None, "strength": "none",
            "fill_rate": round(len(observations) / group_pages, 3),
            "top_labels": [label for label, _ in labels.most_common(3)]}
    located = [o for o in observations if o["point"] is not None]
    if not located:
        return rule
    points = np.array([o["point"] for o in located], dtype=float)
    rng, peak, spread = range_around_peak(points, cfg)
    _, support = density_peak(points, cfg.bbox_tolerance)
    near_peak = [o["bbox"] for o, p in zip(located, points) if (np.abs(p - peak) <= cfg.bbox_tolerance).all()]
    n = len(points)
    hits, trials = (leave_one_out_hits(points, cfg), n) if n >= 2 else (0, 0)
    consolidated = support / n >= cfg.min_support_fraction and support >= cfg.min_support_docs
    rule.update({
        "n_boxes": n, "support": support, "loo_hits": int(hits),
        "bbox_center": [round(float(v), 4) for v in peak],
        "bbox_median": [round(float(np.median([b[j] for b in near_peak])), 4) for j in range(4)],
        "spread": round(spread, 4), "range": [round(v, 4) for v in rng],
        "reliability": round((hits + alpha) / (trials + alpha + beta), 4),
        "strength": ("strong" if support >= cfg.strong_support_docs else "weak") if consolidated else
                    "single" if group_pages == 1 else "none",
    })
    return rule


def fit_reliability_prior(cohort_points: list[list[list[float]]], cfg: Config) -> tuple[float, float, float, int]:
    """Empirical-Bayes Beta prior for reliability: (alpha, beta, p0, cohorts used). p0 is how often one box centre
    of an entity lies within box_range_min of another centre of the same entity, over pages of one cohort; the
    prior counts as `rule_prior_strength` pages."""
    m, rates = cfg.box_range_min, []
    for points in cohort_points:
        if len(points) >= 2:
            rates.append(np.mean([abs(a[0] - b[0]) <= m and abs(a[1] - b[1]) <= m
                                  for i, a in enumerate(points) for j, b in enumerate(points) if i != j]))
    p0 = float(np.mean(rates)) if rates else 0.5
    s = cfg.rule_prior_strength
    return s * p0, s * (1 - p0), p0, len(rates)
