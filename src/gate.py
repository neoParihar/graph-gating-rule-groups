"""From the rule-group posterior to a confidence per rule, and the gate.

    conf_k = P(c_k ∈ R_{G*,k} | e) = Σ_G P(G | e) · P(c_k ∈ R_{G*,k} | G)

| confidence           | what the model is shown                          |
|----------------------|--------------------------------------------------|
| >= gate_box          | the rule's typical box and its confidence        |
| gate_label..gate_box | label hint only, "position uncertain"            |
| < gate_label         | nothing: the rule is withheld                    |
"""

from __future__ import annotations

import pandas as pd

import helpers as H
from src.data import NEW
from src.posterior import RuleGroupPosterior
from src.rule_memory import RuleMemory
from src.schema import BOX_KEYS, FEATURES, page_type

GATES = ("box", "label", "withheld")


class Gate:
    def __init__(self, posterior: RuleGroupPosterior, memory: RuleMemory):
        self.post = posterior
        self.memory = memory

    @property
    def cfg(self):
        return self.post.cfg

    def with_posterior(self, posterior: RuleGroupPosterior) -> "Gate":
        return Gate(posterior, self.memory)

    def in_range_probability(self, group_id: str, key: str, chosen: str) -> float:
        """P(c_k ∈ R_{G*,k} | G): how likely the value sits in the chosen group's range if the page really is from
        `group_id`."""
        rule = self.post.groups[chosen]["rules"][key]
        r_star, a_star = rule["range"], H.range_area(rule["range"])
        if group_id == chosen:
            return rule["reliability"]                                 # the rule's own reliability
        other = None if group_id == NEW else self.post.groups[group_id]["rules"].get(key)
        if other is None:
            return a_star                                              # a value placed uniformly lands in R* by its area
        a, inter = H.range_area(other["range"]), H.range_overlap_area(other["range"], r_star)
        return other["reliability"] * inter / a + (1 - other["reliability"]) * (a_star - inter) / (1 - a)

    def rule_confidence(self, probabilities: dict[str, float], key: str, chosen: str) -> float | None:
        """Σ_G P(G | e) P(c_k ∈ R_{G*,k} | G), by the law of total probability. None when G* has no range for k."""
        if key not in self.post.groups[chosen]["rules"]:
            return None
        return sum(p * self.in_range_probability(group_id, key, chosen) for group_id, p in probabilities.items())

    def confidence_breakdown(self, page: dict, vendor_name: str, key: str, top: int = 5) -> pd.DataFrame:
        """The sum behind one rule's confidence, term by term: for the most probable groups, P(G | e), the chance the
        value is in the chosen group's range if the page were from G, and their product. The rest is lumped."""
        log_terms = self.post.evidence_log_terms(page, vendor_name)
        probabilities = self.post.group_probabilities(log_terms)
        chosen = max(probabilities, key=probabilities.get)
        if chosen == NEW or key not in self.post.groups[chosen]["rules"]:
            return pd.DataFrame()
        rows = [{"group": i, "vendor": log_terms.at[i, "vendor"], "is_chosen": i == chosen, "p_group": p,
                 "p_in_range_if_group": self.in_range_probability(i, key, chosen)} for i, p in probabilities.items()]
        df = pd.DataFrame(rows).sort_values("p_group", ascending=False)
        df["contribution"] = df.p_group * df.p_in_range_if_group
        head, rest = df.head(top), df.iloc[top:]
        if not rest.empty:
            head = pd.concat([head, pd.DataFrame([{"group": f"{len(rest)} others", "vendor": "", "is_chosen": False,
                                                   "p_group": rest.p_group.sum(), "p_in_range_if_group": None,
                                                   "contribution": rest.contribution.sum()}])])
        return head.reset_index(drop=True)

    def decision(self, confidence: float | None) -> str:
        if confidence is None:
            return "no_position"
        return "box" if confidence >= self.cfg.gate_box else "label" if confidence >= self.cfg.gate_label else \
            "withheld"

    def page_guide(self, page: dict, vendor_name: str) -> dict:
        """The gated guide for one page: the posterior summary, the chosen (most probable) group, and every rule of
        it with its confidence and gate decision (withheld rules included, so they can be evaluated)."""
        pt = page_type(page)
        log_terms = self.post.evidence_log_terms(page, vendor_name)
        probabilities = self.post.group_probabilities(log_terms)
        chosen = max(probabilities, key=probabilities.get)
        p_filled = sum(p * self.post.smoothed_fill_rates(NEW if i == NEW else self.post.groups[i], pt)
                       for i, p in probabilities.items())
        guide = {"page": page["number"], "vendor_key": H.vendor_key(vendor_name), "chosen": chosen,
                 "chosen_vendor": log_terms.at[chosen, "vendor"], "p_chosen": probabilities[chosen],
                 "p_new": probabilities[NEW],
                 "vendor_probabilities": self.post.vendor_probabilities(log_terms, probabilities),
                 "top_groups": dict(sorted(probabilities.items(), key=lambda kv: -kv[1])[:5]),
                 "rules": [], "tables": []}
        if chosen == NEW:
            return guide
        G = self.post.groups[chosen]
        guide["chosen_pages"] = G["pages"]
        for key in BOX_KEYS:
            rule = G["rules"].get(key, {})
            confidence = self.rule_confidence(probabilities, key, chosen)
            guide["rules"].append({"name": key, "confidence": confidence, "gate": self.decision(confidence),
                                   "p_filled": float(p_filled[FEATURES.index(key)]), "range": rule.get("range"),
                                   "bbox_median": rule.get("bbox_median"), "reliability": rule.get("reliability"),
                                   "strength": rule.get("strength", "none"), "labels": G["labels"].get(key) or []})
        for table in H.TABLE_TYPES:
            guide["tables"].append({"name": table, "p_present": float(p_filled[FEATURES.index(table)])})
        return guide

    def guide_text(self, guide: dict) -> str:
        """A page guide as prompt text: only rules that passed the gate, each with its probabilities."""
        lines = [f"PAGE {guide['page']}"]
        if guide["chosen"] == NEW:
            lines.append(f"  No remembered layout fits this page (P(unseen layout) = {guide['p_new']:.2f}); no "
                         "vendor-specific rules.")
        else:
            lines.append(f"  Layout from vendor '{guide['chosen_vendor']}' (P(this layout | page) = "
                         f"{guide['p_chosen']:.2f}, built from {guide['chosen_pages']} invoice page(s)):")
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
            columns = {r.name: r.top_labels for r in self.memory.group_path_rules(guide["chosen"]).itertuples()
                       if r.kind == "table"}
            for t in guide["tables"]:
                if t["p_present"] >= 0.5:
                    lines.append(f"    TABLE {t['name']}: present with probability {t['p_present']:.2f}; usually "
                                 f"filled columns: {', '.join(columns.get(t['name'], [])) or '-'}")
        corpus = self.post.corpus_views[page_type({"number": guide["page"]})]["labels"]
        lines.append("  Labels common across all vendors:")
        lines += [f"    {k}: {', '.join(repr(l) for l in corpus[k])}" for k in BOX_KEYS if corpus.get(k)]
        return "\n".join(lines)
