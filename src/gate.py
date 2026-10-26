"""From the layout posterior to a confidence per rule, and the gate (gating.ipynb, "From the layout posterior to a
confidence for each rule").

    conf_k = P(c_k ∈ R_{L*,k} | e) = Σ_L P(L | e) · P(c_k ∈ R_{L*,k} | L)

| confidence           | what the model is shown                         |
|----------------------|-------------------------------------------------|
| >= gate_box          | the rule's typical box and its confidence        |
| gate_label..gate_box | label hint only, "position uncertain"            |
| < gate_label         | nothing: the rule is withheld                    |
"""

from __future__ import annotations

import pandas as pd

import helpers as H
from src.data import NEW
from src.memory_graph import BOX_KEYS, FEATURES, MemoryGraph
from src.posterior import LayoutPosterior

GATES = ("box", "label", "withheld")


class Gate:
    def __init__(self, posterior: LayoutPosterior, graph: MemoryGraph):
        self.post = posterior
        self.graph = graph

    @property
    def cfg(self):
        return self.post.cfg

    def with_posterior(self, posterior: LayoutPosterior) -> "Gate":
        return Gate(posterior, self.graph)

    def hit_given_layout(self, layout_id: str, key: str, chosen: str) -> float:
        """P(c_k ∈ R_{L*,k} | L): how likely the value sits in the shown range if the page really is `layout_id`."""
        rule = self.post.vendor_layouts[chosen]["rules"][key]
        r_star, a_star = rule["range"], H.range_area(rule["range"])
        if layout_id == chosen:
            return rule["q"]                                   # the rule's own reliability
        other = None if layout_id == NEW else self.post.vendor_layouts[layout_id]["rules"].get(key)
        if other is None:
            return a_star                                      # a value placed uniformly lands in R* by its area
        a, inter = H.range_area(other["range"]), H.range_overlap(other["range"], r_star)
        return other["q"] * inter / a + (1 - other["q"]) * (a_star - inter) / (1 - a)

    def rule_confidence(self, post: dict[str, float], key: str, chosen: str) -> float | None:
        """Σ_L P(L | e) P(c_k ∈ R_{L*,k} | L), by the law of total probability. None when L* has no range for k."""
        if key not in self.post.vendor_layouts[chosen]["rules"]:
            return None
        return sum(p * self.hit_given_layout(layout_id, key, chosen) for layout_id, p in post.items())

    def confidence_breakdown(self, page: dict, vendor_name: str, key: str, top: int = 5) -> pd.DataFrame:
        """The sum behind one rule's confidence, term by term: for the most probable layouts, P(L | e), the chance
        the value is in the MAP layout's range if the page were L, and their product. The rest is lumped."""
        scores = self.post.scores(page, vendor_name)
        post = self.post.posterior(scores)
        chosen = max(post, key=post.get)
        if chosen == NEW or key not in self.post.vendor_layouts[chosen]["rules"]:
            return pd.DataFrame()
        rows = [{"layout": i, "vendor": scores.at[i, "vendor"], "is_map": i == chosen, "p_layout": p,
                 "p_in_range_if_layout": self.hit_given_layout(i, key, chosen)} for i, p in post.items()]
        df = pd.DataFrame(rows).sort_values("p_layout", ascending=False)
        df["contribution"] = df.p_layout * df.p_in_range_if_layout
        head, rest = df.head(top), df.iloc[top:]
        if not rest.empty:
            head = pd.concat([head, pd.DataFrame([{"layout": f"{len(rest)} others", "vendor": "", "is_map": False,
                                                   "p_layout": rest.p_layout.sum(),
                                                   "p_in_range_if_layout": None,
                                                   "contribution": rest.contribution.sum()}])])
        return head.reset_index(drop=True)

    def decide(self, conf: float | None) -> str:
        if conf is None:
            return "no_position"
        return "box" if conf >= self.cfg.gate_box else "label" if conf >= self.cfg.gate_label else "withheld"

    def page_rules(self, page: dict, vendor_name: str) -> dict:
        """The gated guide for one page: posterior summary, the MAP layout, and every rule of it with its confidence
        and gate decision (withheld rules included, so they can be evaluated)."""
        pn = min(page["number"], 2)
        scores = self.post.scores(page, vendor_name)
        post = self.post.posterior(scores)
        chosen = max(post, key=post.get)
        vp = self.post.vendor_posterior(scores, post)
        p_filled = sum(p * self.post.theta(NEW if i == NEW else self.post.vendor_layouts[i], pn)
                       for i, p in post.items())
        out = {"page": page["number"], "vendor_key": H.vendor_key(vendor_name), "chosen": chosen,
               "chosen_vendor": scores.at[chosen, "vendor"], "p_chosen": post[chosen], "p_new": post[NEW],
               "vendor_posterior": vp, "top_layouts": dict(sorted(post.items(), key=lambda kv: -kv[1])[:5]),
               "rules": [], "tables": []}
        if chosen == NEW:
            return out
        L = self.post.vendor_layouts[chosen]
        out["chosen_docs"] = L["docs"]
        for key in BOX_KEYS:
            rule = L["rules"].get(key, {})
            conf = self.rule_confidence(post, key, chosen)
            out["rules"].append({"name": key, "confidence": conf, "gate": self.decide(conf),
                                 "p_filled": float(p_filled[FEATURES.index(key)]), "range": rule.get("range"),
                                 "bbox_median": rule.get("bbox_median"), "q": rule.get("q"),
                                 "strength": rule.get("strength", "none"), "labels": L["labels"].get(key) or []})
        for table in H.TABLE_TYPES:
            out["tables"].append({"name": table, "p_present": float(p_filled[FEATURES.index(table)])})
        return out

    def rules_to_text(self, guide: dict) -> str:
        """Render a gated guide as prompt text: only rules that passed the gate, each with its probabilities."""
        lines = [f"PAGE {guide['page']}"]
        if guide["chosen"] == NEW:
            lines.append(f"  No remembered layout fits this page (P(unseen layout) = {guide['p_new']:.2f}); no "
                         "vendor-specific rules.")
        else:
            lines.append(f"  Layout from vendor tree '{guide['chosen_vendor']}' (P(this layout | page) = "
                         f"{guide['p_chosen']:.2f}, built from {guide['chosen_docs']} invoice(s)):")
            for r in guide["rules"]:
                labels = f"; labelled {', '.join(repr(l) for l in r['labels'])}" if r["labels"] else ""
                if r["gate"] == "box":
                    where = f"value usually at box_2d {H.bbox_to_box2d(r['bbox_median'])} " \
                            f"(confidence {r['confidence']:.2f})"
                elif r["gate"] == "label":
                    where = f"position uncertain (confidence {r['confidence']:.2f})"
                elif r["gate"] == "no_position" and labels:
                    where = "position not remembered"
                else:
                    continue
                lines.append(f"    {r['name']}: expected with probability {r['p_filled']:.2f}; {where}{labels}")
            columns = {r.name: r.top_labels for r in self.graph.layout_rules(guide["chosen"]).itertuples()
                       if r.kind == "table"}
            for t in guide["tables"]:
                if t["p_present"] >= 0.5:
                    lines.append(f"    TABLE {t['name']}: present with probability {t['p_present']:.2f}; usually "
                                 f"filled columns: {', '.join(columns.get(t['name'], [])) or '-'}")
        corpus = self.post.corpus_layouts[min(guide["page"], 2)]["labels"]
        lines.append("  Labels common across all vendors:")
        lines += [f"    {k}: {', '.join(repr(l) for l in corpus[k])}" for k in BOX_KEYS if corpus.get(k)]
        return "\n".join(lines)
