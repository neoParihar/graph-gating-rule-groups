"""The rule-group memory graph in Neo4j: documents become paths, similar paths become rule groups.

    (:Vendor)-[:HAS_RULE_GROUP]->(:RuleGroup {document_type})-[:_]->(:EntityRule)-[:_]-> … -[:_]->(:TableRule)
    (:Vendor)-[:HAS_DOCUMENT]->(:Document)-[:HAS_PAGE]->(:Page)-[:_]->(:Field)-[:_]-> … -[:_]->(:Table)
    (:Page)-[:IN_GROUP {role, order, similarity}]->(:RuleGroup)
    (:Field|Table)-[:SUPPORTS]->(:EntityRule|TableRule)
    (:EntityRule|Field)-[:OF_KIND]->(:Entity)-[:NORMALIZED_BY|CONSTRAINED_BY]->(:NormalizationRule|:Invariant)

A page path holds only what the page has (src/schema.py). The `_` edges only chain a path together; every edge
that means something has a name. The Cypher lives in src/queries/.

Ingestion (the write path): a page and a rule group are both turned into structure vectors (which entities and
tables, and roughly where; a group's is the mean of its pages'). A training page is a candidate for a rule group of
its vendor when the cosine of the two is at least `rule_group_similarity`, and it joins the most similar candidate
whose ranges hold every node the two share. The group's path then grows by any node the page adds, and the group is
re-consolidated (src/consolidation.py). Otherwise the page founds a new group. Choosing a group for a noisy test
page is the posterior's job (src/posterior.py), not this module's.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

import helpers as H
from src.config import Config
from src.consolidation import consolidate_rule, fit_reliability_prior
from src.data import Corpus
from src.resources import load_cypher
from src.schema import BOX_KEYS, CORPUS_VIEW, FEATURES, NODE_ORDER, UNIDENTIFIED, page_path, page_type, \
    path_signature

NODE_CONSTRAINTS = [("Vendor", "key"), ("RuleGroup", "id"), ("EntityRule", "id"), ("TableRule", "id"),
                    ("Document", "doc_id"), ("Page", "id"), ("Field", "id"), ("Table", "id"), ("Entity", "key"),
                    ("NormalizationRule", "name"), ("Invariant", "name")]


class RuleMemory:
    """One Neo4j database holding the rule-group memory. `build` writes it from a corpus's training documents;
    everything else reads it."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password),
                                           notifications_min_severity="OFF")
        self.driver.verify_connectivity()
        self.rule_prior: tuple[float, float, float, int] = (1.0, 1.0, 0.5, 0)   # (alpha, beta, p0, cohorts)
        self._groups: dict[str, dict] | None = None
        self._order = 0

    def close(self) -> None:
        self.driver.close()

    # --- Cypher -----------------------------------------------------------------------------------------------

    def cypher(self, query: str, **params) -> list[dict]:
        records, _, _ = self.driver.execute_query(query, **params)
        return [r.data() for r in records]

    def cypher_df(self, query: str, **params) -> pd.DataFrame:
        return pd.DataFrame(self.cypher(query, **params))

    def run(self, query_name: str, /, **params) -> list[dict]:
        """Run the query in src/queries/<query_name>.cypher."""
        return self.cypher(load_cypher(query_name), **params)

    def run_df(self, query_name: str, /, **params) -> pd.DataFrame:
        return pd.DataFrame(self.run(query_name, **params))

    # --- Build ------------------------------------------------------------------------------------------------

    def build(self, corpus: Corpus, reset: bool = True) -> None:
        """Ingest every training page in doc_id order, then fit the reliability prior and consolidate every group
        with it. With reset=False an already-built memory for the same split is kept as it is."""
        if not reset and self.training_doc_ids() == corpus.train_ids:
            self.rule_prior = self.estimate_reliability_prior()
            self._groups = None
            return
        self.cypher("MATCH (n) DETACH DELETE n")
        for label, prop in NODE_CONSTRAINTS:
            self.cypher(f"CREATE CONSTRAINT rm_{label.lower()}_{prop} IF NOT EXISTS "
                        f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")
        self._write_dictionary()

        self._order = 0
        for doc_id in sorted(corpus.train_ids):
            rec = corpus.records[doc_id]
            for page in rec["pages"]:
                self.ingest_page(rec, page)
        self.rule_prior = self.estimate_reliability_prior()
        for r in self.cypher("MATCH (g:RuleGroup) RETURN g.id AS id"):
            self.consolidate_group(r["id"])
        self._groups = None
        self._assert_consistent(corpus)

    def _write_dictionary(self) -> None:
        """The nodes every path links to: entity kinds, their normalization rules, the arithmetic invariant, and
        the unidentified vendor."""
        self.cypher("UNWIND $rules AS r MERGE (n:NormalizationRule {name: r.name}) SET n.rule = r.rule",
                    rules=[{"name": k, "rule": v} for k, v in H.NORMALIZATION_RULES.items()])
        self.cypher("MERGE (i:Invariant {name: 'arithmetic_check'}) SET i.expression = $expr",
                    expr=H.ARITHMETIC_INVARIANT)
        self.cypher("UNWIND $keys AS k MERGE (e:Entity {key: k.key}) SET e.kind = k.kind",
                    keys=[{"key": k, "kind": "table" if k in H.TABLE_TYPES else "entity"} for k in FEATURES])
        self.run("link_entity_normalization", types={k: H.VALUE_TYPES[k] for k in BOX_KEYS})
        self.run("link_entity_invariant", members=sorted(H.INVARIANT_MEMBERS | {
            t for t, cols in H.TABLE_TYPES.items() if set(cols) & H.INVARIANT_MEMBERS}))
        self.cypher("MERGE (v:Vendor {key: $key}) SET v.names = ['Unidentified']", key=UNIDENTIFIED)

    def ingest_page(self, rec: dict, page: dict) -> dict:
        """Write one training page and place it: join the most similar rule group whose ranges hold it, or found a
        new one. Returns the decision (the ingestion log row)."""
        vendor = rec["vendor_key"] or UNIDENTIFIED
        path = page_path(page)
        pt, structure = page_type(page), self.structure_vector_of(page)
        page_id = f"{rec['doc_id']}|p{page['number']}"
        if self.cypher("MATCH (p:Page {id: $id}) RETURN p", id=page_id):
            return {}                                                  # already ingested: idempotent
        if vendor != UNIDENTIFIED:
            self.run("merge_vendor", key=vendor, name=rec["vendor_name"])
        self.run("write_document", vendor=vendor, doc_id=rec["doc_id"], doc_type=H.DOC_TYPE)
        self.run("write_page", doc_id=rec["doc_id"], page_id=page_id, number=page["number"], page_type=pt,
                 signature=path_signature(path), structure=structure,
                 nodes=[{**n, "id": f"{page_id}|{n['key']}"} for n in path])

        candidates = self.matching_groups(vendor, pt, path, structure)
        passed = candidates[candidates.passed]
        if not passed.empty:
            best = passed.sort_values(["similarity", "n_pages"], ascending=False).iloc[0]
            group_id, role, similarity = best.group, "joined", round(float(best.similarity), 4)
            have = {r["key"] for r in self.cypher("MATCH (:RuleGroup {id: $id})-[:`_`*]->(r) RETURN r.key AS key",
                                                  id=group_id)}
        else:
            n = self.cypher("MATCH (:Vendor {key: $v})-[:HAS_RULE_GROUP]->(g:RuleGroup {page: $pt}) "
                            "RETURN count(g) AS n", v=vendor, pt=pt)[0]["n"]
            group_id, role, similarity, have = f"{vendor}|p{pt}|g{n}", "founded", None, set()
            self.run("found_group", vendor=vendor, group_id=group_id, doc_type=H.DOC_TYPE, page_type=pt,
                     zeros=[0.0] * len(structure))
        added = [{"kind": n["kind"], "key": n["key"]} for n in path if n["key"] not in have]
        if added or role == "founded":
            self.run("add_rules", group_id=group_id, nodes=added)
            keys = sorted(have | {n["key"] for n in added}, key=NODE_ORDER.get)
            self.run("unchain_group", group_id=group_id)
            self.run("rechain_group", group_id=group_id, ids=[f"{group_id}|{k}" for k in keys],
                     signature=" > ".join(keys))
        self._order += 1
        self.run("join_group", group_id=group_id, page_id=page_id, role=role, order=self._order,
                 candidates=len(candidates), passed=len(passed), similarity=similarity, structure=structure)
        self.consolidate_group(group_id)
        self._groups = None
        return {"doc_id": rec["doc_id"], "page": page["number"], "group": group_id, "role": role,
                "candidates": len(candidates), "passed": len(passed), "similarity": similarity}

    def structure_vector_of(self, page: dict) -> list[float]:
        return H.structure_vector(page, self.cfg.position_grid, self.cfg.position_weight)

    def matching_groups(self, vendor: str, pt: int, path: list[dict], structure: list[float]) -> pd.DataFrame:
        """The ingestion rule, read-only: the vendor's rule groups whose structure vector has cosine >=
        rule_group_similarity with the page's, and for each whether every node the two share lies inside the
        group's range for it (`passed`)."""
        cols = ["group", "n_pages", "similarity", "checked", "inside", "outside", "passed"]
        groups = pd.DataFrame([{**r, "similarity": H.cosine_similarity(r["structure_vector"], structure)}
                               for r in self.run("candidate_groups", vendor=vendor, doc_type=H.DOC_TYPE,
                                                 page_type=pt)],
                              columns=["group", "n_pages", "structure_vector", "similarity"])
        groups = groups[groups.similarity >= self.cfg.rule_group_similarity].drop(columns="structure_vector")
        if groups.empty:
            return pd.DataFrame(columns=cols).astype({"passed": bool})
        checked = self.run_df("range_check", groups=groups.group.tolist(),
                              nodes=[{"key": n["key"], "point": n["point"]} for n in path])
        df = groups.merge(checked, on="group", how="left") if not checked.empty else \
            groups.assign(checked=0, inside=0, outside=None)
        df[["checked", "inside"]] = df[["checked", "inside"]].fillna(0).astype(int)
        df["outside"] = df["outside"].map(lambda o: o if isinstance(o, list) else [])
        df["passed"] = df.inside == df.checked
        return df[cols]

    def ingestion_match(self, page: dict, vendor_name: str) -> str | None:
        """The ingestion rule applied to a page that is not written: the group it would join, or None when it would
        found a new one. For comparison with the gate on noisy test pages."""
        vendor = H.vendor_key(vendor_name) or UNIDENTIFIED
        candidates = self.matching_groups(vendor, page_type(page), page_path(page), self.structure_vector_of(page))
        passed = candidates[candidates.passed]
        return None if passed.empty else passed.sort_values(["similarity", "n_pages"], ascending=False).group.iloc[0]

    # --- Consolidation ----------------------------------------------------------------------------------------

    def consolidate_group(self, group_id: str) -> None:
        """Rebuild every rule node of one group from all the observations that support it."""
        n_pages = self.cypher("MATCH (g:RuleGroup {id: $id}) RETURN g.n_pages AS n", id=group_id)[0]["n"]
        by_rule = defaultdict(list)
        for o in self.run("group_observations", group_id=group_id):
            by_rule[(o["rule"], o["kind"])].append(o)
        rules = []
        for (rule_id, kind), obs in by_rule.items():                    # every rule node has >= 1 supporting page
            props = consolidate_rule(obs, n_pages, self.cfg, self.rule_prior[:2])
            if kind == "TableRule":
                cols = Counter(c for o in obs for c in (o["filled_columns"] or []))
                props["mean_rows"] = round(float(np.mean([o["rows"] for o in obs])), 1)
                props["columns"] = [c for c in H.TABLE_TYPES[obs[0]["key"]] if cols[c] / len(obs) >= 0.5]
                props["region_median"], props["bbox_median"] = props["bbox_median"], None
            rules.append({"id": rule_id, "props": props})
        self.run("set_rules", rules=rules)

    def estimate_reliability_prior(self) -> tuple[float, float, float, int]:
        """The empirical-Bayes Beta prior for reliability, from (vendor, page type, structure) cohorts of the
        stored document paths: (alpha, beta, p0, cohorts used)."""
        return fit_reliability_prior([r["points"] for r in self.run("prior_cohorts")], self.cfg)

    def _assert_consistent(self, corpus: Corpus) -> None:
        assert self.training_doc_ids() == corpus.train_ids, "a training document is missing or duplicated"
        n_pages = sum(len(corpus.records[d]["pages"]) for d in corpus.train_ids)
        placed = self.cypher("MATCH (p:Page) OPTIONAL MATCH (p)-[m:IN_GROUP]->() RETURN p.id AS id, count(m) AS n")
        assert len(placed) == n_pages and all(r["n"] == 1 for r in placed), "a page is not in exactly one group"
        bad = self.run("check_unsupported_page_nodes")
        assert not bad, f"a page node supports no rule of its group: {bad[:3]}"
        bad = self.run("check_unsupported_rules")
        assert not bad, f"a rule with no supporting page: {bad[:3]}"
        bad = self.run("check_path_signatures")
        assert not bad, f"a group path out of step with its signature: {bad[:3]}"
        counts = self.run("check_founding_pages")
        assert all(r["founded"] == 1 for r in counts), "a group without exactly one founding page"
        assert sum(r["n"] for r in counts) == n_pages

    # --- Reading ----------------------------------------------------------------------------------------------

    def node_counts(self) -> pd.DataFrame:
        return self.cypher_df("MATCH (n) RETURN labels(n)[0] AS node, count(*) AS count ORDER BY node")

    def training_doc_ids(self) -> set[str]:
        return {r["d"] for r in self.cypher("MATCH (d:Document) RETURN d.doc_id AS d")}

    def vendor_names(self) -> dict[str, list[str]]:
        """{vendor key: the names printed for it}, for every known vendor."""
        return {r["vendor"]: r["names"] for r in self.run("vendor_names", unidentified=UNIDENTIFIED)}

    @property
    def p_new(self) -> float:
        """Good–Turing mass of an unseen group: groups founded by a single training page ÷ training pages. The
        observed chance that the next page founds a new group."""
        r = self.cypher("MATCH (g:RuleGroup) RETURN sum(CASE WHEN g.n_pages = 1 THEN 1 ELSE 0 END) AS n1, "
                        "sum(g.n_pages) AS n")[0]
        return r["n1"] / r["n"]

    def rule_groups(self) -> dict[str, dict]:
        """Every rule group, plus the corpus-wide view per page type, read once in the shape the posterior uses:
        {id: {id, vendor, page, pages, filled_pages: array over FEATURES, rules: {key: {range, reliability,
        bbox_median, strength}}, labels: {key: top labels}}}. `filled_pages` counts the group's pages that have the
        node; the corpus views have vendor CORPUS_VIEW and no rules."""
        if self._groups is None:
            groups = {}
            for r in self.run("group_stats"):
                G = groups.setdefault(r["group"], {"id": r["group"], "vendor": r["vendor"], "page": r["page"],
                                                   "pages": r["n_pages"], "filled_pages": {}, "rules": {},
                                                   "labels": {}})
                if r["key"] is None:
                    continue
                G["filled_pages"][r["key"]] = r["n_supported"]
                G["labels"][r["key"]] = r["top_labels"] or []
                if r["kind"] == "EntityRule" and r["range"]:
                    G["rules"][r["key"]] = {"range": r["range"], "reliability": r["reliability"],
                                            "bbox_median": r["bbox_median"], "strength": r["strength"]}
            views = {}
            for r in self.run("corpus_stats"):
                V = views.setdefault(r["page"], {"id": f"{CORPUS_VIEW}|p{r['page']}", "vendor": CORPUS_VIEW,
                                                 "page": r["page"], "pages": r["pages"], "filled_pages": {},
                                                 "rules": {}, "labels": {}})
                if r["key"] is not None:
                    V["filled_pages"][r["key"]] = r["n_filled"]
                    V["labels"][r["key"]] = [label for label, _ in Counter(r["labels"]).most_common(3)]
            groups.update({V["id"]: V for V in views.values()})
            for G in groups.values():
                G["filled_pages"] = np.array([G["filled_pages"].get(f, 0) for f in FEATURES], dtype=float)
            self._groups = groups
        return self._groups

    def ingestion_log(self, vendor: str | None = None) -> pd.DataFrame:
        """Where every training page went, in ingestion order: the group, whether the page founded or joined it,
        how many of the vendor's groups were similar enough to be candidates and passed the range check, and the
        cosine with the group it joined."""
        return self.run_df("ingestion_log", vendor=vendor)

    def groups_table(self) -> pd.DataFrame:
        """One row per rule group: vendor, page type, pages, nodes on its path, and its structure."""
        return self.run_df("groups_table")

    def rules_table(self) -> pd.DataFrame:
        """One row per entity rule with a range: the consolidated statistics."""
        return self.run_df("rules_table")

    def rule_observations(self, group_id: str, key: str) -> dict:
        """One rule node's consolidated properties, and the observations that support it (`observations`)."""
        rule = self.run("rule_node", id=f"{group_id}|{key}")[0]
        obs = [o for o in self.run("group_observations", group_id=group_id) if o["key"] == key]
        rule["observations"] = pd.DataFrame([{"doc_id": o["doc_id"], "point": o["point"], "bbox": o["bbox"],
                                              "label": o["label"]} for o in obs])
        return rule

    def group_path_rules(self, group_id: str) -> pd.DataFrame:
        """Every node of one group's path, in path order: entities with their rule, tables with their columns."""
        rules = self.run_df("group_path_rules", id=group_id)
        if not rules.empty:
            rules["top_labels"] = rules["top_labels"].map(lambda l: l if isinstance(l, list) else [])
        return rules

    def similar_group_pairs(self, min_similarity: float) -> list[dict]:
        """Rule groups of different vendors on the same page type with similar structure vectors."""
        return self.run("similar_group_pairs", min=min_similarity)
