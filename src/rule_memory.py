"""The rule-group memory graph in Neo4j: documents become paths, similar paths become rule groups.

    (:Vendor)-[:HAS_RULE_GROUP]->(:RuleGroup {document_type})-[:_]->(:EntityRule)-[:_]-> … -[:_]->(:TableRule)
    (:Vendor)-[:HAS_DOCUMENT]->(:Document)-[:HAS_PAGE]->(:Page)-[:_]->(:Field)-[:_]-> … -[:_]->(:Table)
    (:Page)-[:IN_GROUP {role, order, similarity}]->(:RuleGroup)
    (:Field|Table)-[:SUPPORTS]->(:EntityRule|TableRule)
    (:EntityRule|Field)-[:OF_KIND]->(:Entity)-[:NORMALIZED_BY|CONSTRAINED_BY]->(:NormalizationRule|:Invariant)

A page path holds only what the page has: one node per filled entity and per table, in schema order. The `_`
edges only chain a path together; every edge that means something has a name.

Ingestion (the write path): a page and a rule group are both turned into structure vectors (which entities and
tables, and roughly where: `H.structure_vector`; a group's is the mean of its pages'). A training page is a candidate
for a rule group of its vendor when the cosine of the two is at least `rule_group_similarity`, and it joins the most
similar candidate whose ranges hold every node the two share. The group's path then grows by any node the page
adds. Otherwise the page founds a new group. Choosing a group for a noisy test page is the posterior's job
(src/posterior.py), not this module's.

`layouts()` hands the rule groups to the posterior and the gate in the same shape `MemoryGraph.layouts()` does,
so the confidence math runs unchanged on either memory.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

import helpers as H
from src.config import Config
from src.data import Corpus
from src.memory_graph import ALL_VENDORS, BOX_KEYS, FEATURES, table_payload

UNIDENTIFIED = H.UNCLASSIFIED          # vendor key of pages printed without a vendor name
NODE_ORDER = {key: i for i, key in enumerate(FEATURES)}   # schema order: entities (JSON order), then tables


# --- A page as a path ---------------------------------------------------------------------------------------------


def page_type(page: dict) -> int:
    """1 for a first page, 2 for every continuation page."""
    return min(page["number"], 2)


def page_path(page: dict) -> list[dict]:
    """The page's path: one node per filled entity and per table with rows, in schema order. `point` is what the
    range check tests: an entity's box centre, a table's top-left corner (a table grows downward with its rows).
    None when the value has no usable box."""
    nodes = []
    for key in BOX_KEYS:
        f = page["fields"].get(key)
        if f and f["value"]:
            nodes.append({"kind": "entity", "key": key, "bbox": f["bbox"], "label": f.get("label", ""),
                          "point": list(H.bbox_center(f["bbox"])) if f["bbox"] else None})
    for t in table_payload(page):
        if t["rows"]:
            region = t["region"] if t["region"][0] >= 0 else None
            nodes.append({"kind": "table", "key": t["type"], "bbox": region, "rows": t["rows"], "label": "",
                          "point": region[:2] if region else None,
                          "filled_columns": [c["header"] for c in t["columns"] if any(c["filled"])]})
    return sorted(nodes, key=lambda n: NODE_ORDER[n["key"]])


def signature(path: list[dict]) -> str:
    """The path's structure: its node types in order."""
    return " > ".join(n["key"] for n in path)


# --- Consolidation of one rule node -------------------------------------------------------------------------------


def _peak(points: np.ndarray, tol: float) -> tuple[int, int]:
    """(index of the point with the most neighbours within tol, that count)."""
    near = (np.abs(points[:, None, :] - points[None, :, :]) <= tol).all(axis=2).sum(axis=1)
    i = int(np.argmax(near))
    return i, int(near[i])


def _range(points: np.ndarray, cfg: Config) -> tuple[list[float], np.ndarray, float]:
    """(range, peak, spread) of a set of points: a square around the density peak, half-width
    clip(k · median distance to the peak, min, max)."""
    i, _ = _peak(points, cfg.bbox_tolerance)
    centre = points[i]
    spread = float(np.median(np.hypot(*(points - centre).T)))
    margin = min(max(cfg.box_range_spread_k * spread, cfg.box_range_min), cfg.box_range_max)
    return H.box_range(*centre, margin), centre, spread


def consolidate_points(obs: list[dict], n_pages: int, cfg: Config, prior: tuple[float, float]) -> dict:
    """The rule of one node of a group from its supporting observations (each {point, bbox, label}).

    reliability is leave-one-out: every observed point is tested against the range rebuilt from the others. A page
    joins a group only when its points are inside the group's ranges, so counting members inside the range would
    overstate the rule; the left-out point never helped draw the range it is tested against."""
    alpha, beta = prior
    labels = Counter(o["label"] for o in obs if o["label"])
    rule = {"n_boxes": 0, "support": 0, "loo_hits": 0, "bbox_center": None, "bbox_median": None, "spread": None,
            "range": None, "reliability": None, "strength": "none", "fill_rate": round(len(obs) / n_pages, 3),
            "top_labels": [label for label, _ in labels.most_common(3)]}
    located = [o for o in obs if o["point"] is not None]
    if not located:
        return rule
    points = np.array([o["point"] for o in located], dtype=float)
    rng, centre, spread = _range(points, cfg)
    _, support = _peak(points, cfg.bbox_tolerance)
    members = [o["bbox"] for o, p in zip(located, points) if (np.abs(p - centre) <= cfg.bbox_tolerance).all()]
    n = len(points)
    hits = sum(H.in_range(tuple(points[i]), _range(np.delete(points, i, axis=0), cfg)[0]) for i in range(n)) \
        if n >= 2 else 0
    trials = n if n >= 2 else 0
    consolidated = support / n >= cfg.min_support_fraction and support >= cfg.min_support_docs
    rule.update({
        "n_boxes": n, "support": support, "loo_hits": int(hits),
        "bbox_center": [round(float(v), 4) for v in centre],
        "bbox_median": [round(float(np.median([b[j] for b in members])), 4) for j in range(4)],
        "spread": round(spread, 4), "range": [round(v, 4) for v in rng],
        "reliability": round((hits + alpha) / (trials + alpha + beta), 4),
        "strength": ("strong" if support >= cfg.strong_support_docs else "weak") if consolidated else
                    "single" if n_pages == 1 else "none",
    })
    return rule


# --- Cypher -------------------------------------------------------------------------------------------------------

WRITE_DOCUMENT = """
MATCH (v:Vendor {key: $vendor})
MERGE (v)-[:HAS_DOCUMENT]->(d:Document {doc_id: $doc_id})
SET d.document_type = $doc_type
"""

WRITE_PAGE = """
MATCH (d:Document {doc_id: $doc_id})
MERGE (d)-[:HAS_PAGE]->(p:Page {id: $page_id})
SET p.number = $number, p.page_type = $page_type, p.signature = $signature, p.structure_vector = $structure
WITH p
UNWIND range(0, size($nodes) - 1) AS i
WITH p, i, $nodes[i] AS n
CALL (n) {
  WITH n WHERE n.kind = 'entity'
  CREATE (f:Field {id: n.id, key: n.key, bbox: n.bbox, point: n.point, label: n.label})
  RETURN f AS node
  UNION
  WITH n WHERE n.kind = 'table'
  CREATE (f:Table {id: n.id, key: n.key, region: n.bbox, point: n.point, rows: n.rows,
                   filled_columns: n.filled_columns})
  RETURN f AS node
}
WITH p, i, node ORDER BY i
WITH p, collect(node) AS chain
FOREACH (j IN range(0, size(chain) - 1) |
  FOREACH (a IN CASE WHEN j = 0 THEN [p] ELSE [chain[j - 1]] END | FOREACH (b IN [chain[j]] | MERGE (a)-[:`_`]->(b))))
WITH chain
UNWIND chain AS node
MATCH (e:Entity {key: node.key})
MERGE (node)-[:OF_KIND]->(e)
"""

CANDIDATE_GROUPS = """
MATCH (:Vendor {key: $vendor})-[:HAS_RULE_GROUP]->(g:RuleGroup {document_type: $doc_type, page: $page_type})
RETURN g.id AS group, g.n_pages AS n_pages, g.structure_vector AS structure_vector ORDER BY g.id
"""

# The range check, done in the graph: each node of the page against the rule node of the same type on the
# group's path. Only nodes the two share are checked. A node without a usable box, or a rule that has no range
# yet, cannot contradict the group.
RANGE_CHECK = """
UNWIND $groups AS gid
MATCH (g:RuleGroup {id: gid})
UNWIND $nodes AS n
MATCH (g)-[:`_`*]->(r {key: n.key})
WITH g, n, n.point IS NULL OR r.range IS NULL OR
     (r.range[0] <= n.point[0] <= r.range[2] AND r.range[1] <= n.point[1] <= r.range[3]) AS inside
RETURN g.id AS group, count(n) AS checked, sum(CASE WHEN inside THEN 1 ELSE 0 END) AS inside,
       collect(CASE WHEN inside THEN null ELSE n.key END) AS outside
"""

FOUND_GROUP = """
MATCH (v:Vendor {key: $vendor})
CREATE (v)-[:HAS_RULE_GROUP]->(:RuleGroup {id: $group_id, document_type: $doc_type, page: $page_type,
                                           n_pages: 0, structure_sum: $zeros})
"""

# New rule nodes for the nodes a page brings that the group's path doesn't have yet.
ADD_RULES = """
MATCH (g:RuleGroup {id: $group_id})
UNWIND $nodes AS n
CALL (n, g) {
  WITH n, g WHERE n.kind = 'entity'
  CREATE (r:EntityRule {id: g.id + '|' + n.key, key: n.key})
  RETURN r
  UNION
  WITH n, g WHERE n.kind = 'table'
  CREATE (r:TableRule {id: g.id + '|' + n.key, key: n.key})
  RETURN r
}
MATCH (e:Entity {key: r.key})
MERGE (r)-[:OF_KIND]->(e)
"""

# Re-chain a group's path in schema order after it grew, and record its structure.
UNCHAIN = "MATCH (:RuleGroup {id: $group_id})-[:`_`*0..]->()-[e:`_`]->() DELETE e"
RECHAIN = """
MATCH (g:RuleGroup {id: $group_id})
SET g.signature = $signature
WITH g
UNWIND range(0, size($ids) - 1) AS j
CALL (g, j) {
  MATCH (b {id: $ids[j]}) WHERE b:EntityRule OR b:TableRule
  OPTIONAL MATCH (prev {id: $ids[j - 1]}) WHERE j > 0 AND (prev:EntityRule OR prev:TableRule)
  WITH b, CASE WHEN j = 0 THEN g ELSE prev END AS a
  MERGE (a)-[:`_`]->(b)
}
"""

JOIN_GROUP = """
MATCH (g:RuleGroup {id: $group_id}), (p:Page {id: $page_id})
MERGE (p)-[m:IN_GROUP]->(g)
SET m.role = $role, m.order = $order, m.candidates = $candidates, m.passed = $passed, m.similarity = $similarity,
    g.n_pages = g.n_pages + 1,
    g.structure_sum = [i IN range(0, size(g.structure_sum) - 1) | g.structure_sum[i] + $structure[i]]
SET g.structure_vector = [x IN g.structure_sum | x / g.n_pages]
WITH g, p
MATCH (p)-[:`_`*]->(f)
MATCH (g)-[:`_`*]->(r {key: f.key})
MERGE (f)-[:SUPPORTS]->(r)
"""

GROUP_OBSERVATIONS = """
MATCH (g:RuleGroup {id: $group_id})-[:`_`*]->(r)<-[:SUPPORTS]-(f)<-[:`_`*]-(p:Page)-[m:IN_GROUP]->(g)
MATCH (d:Document)-[:HAS_PAGE]->(p)
RETURN r.id AS rule, labels(r)[0] AS kind, r.key AS key, m.order AS order, d.doc_id AS doc_id,
       f.point AS point, coalesce(f.bbox, f.region) AS bbox, coalesce(f.label, '') AS label,
       f.rows AS rows, f.filled_columns AS filled_columns
ORDER BY rule, order
"""

SET_RULES = """
UNWIND $rules AS r
CALL (r) {
  MATCH (x:EntityRule {id: r.id}) SET x += r.props
  UNION
  MATCH (x:TableRule {id: r.id}) SET x += r.props
}
"""

GROUP_STATS = """
MATCH (v:Vendor)-[:HAS_RULE_GROUP]->(g:RuleGroup)
OPTIONAL MATCH (g)-[:`_`*]->(r)
RETURN v.key AS vendor, g.id AS group, g.page AS page, g.n_pages AS n_pages, labels(r)[0] AS kind, r.key AS key,
       size([(r)<-[:SUPPORTS]-() | 1]) AS n_supported, r.range AS range, r.reliability AS reliability,
       r.bbox_median AS bbox_median, r.strength AS strength, r.top_labels AS top_labels
"""

CORPUS_STATS = """
MATCH (p:Page)
WITH p.page_type AS page, count(p) AS pages
OPTIONAL MATCH (:Page {page_type: page})-[:`_`*]->(f)
RETURN page, pages, f.key AS key, count(f) AS n_filled, collect(CASE WHEN f.label <> '' THEN f.label END) AS labels
"""


class RuleMemory:
    """One Neo4j database holding the rule-group memory. `build` writes it from a corpus's training documents;
    everything else reads it. The read API mirrors `MemoryGraph`'s, so the posterior, the gate and the scenario
    generator take either memory."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password),
                                           notifications_min_severity="OFF")
        self.driver.verify_connectivity()
        self.rule_prior: tuple[float, float, float, int] = (1.0, 1.0, 0.5, 0)   # (alpha, beta, p0, cohorts)
        self._layouts: dict[str, dict] | None = None
        self._order = 0

    def close(self) -> None:
        self.driver.close()

    # --- Cypher -----------------------------------------------------------------------------------------------

    def cypher(self, query: str, **params) -> list[dict]:
        records, _, _ = self.driver.execute_query(query, **params)
        return [r.data() for r in records]

    def cypher_df(self, query: str, **params) -> pd.DataFrame:
        return pd.DataFrame(self.cypher(query, **params))

    def node_counts(self) -> pd.DataFrame:
        return self.cypher_df("MATCH (n) RETURN labels(n)[0] AS node, count(*) AS count ORDER BY node")

    def train_doc_ids(self) -> set[str]:
        return {r["d"] for r in self.cypher("MATCH (d:Document) RETURN d.doc_id AS d")}

    # --- Build ------------------------------------------------------------------------------------------------

    def build(self, corpus: Corpus, reset: bool = True) -> None:
        """Ingest every training page in doc_id order, then fit the reliability prior and consolidate every
        group with it. With reset=False an already-built memory for the same split is kept as it is."""
        if not reset and self.train_doc_ids() == corpus.train_ids:
            self.rule_prior = self.estimate_rule_prior()
            self._layouts = None
            return
        self.cypher("MATCH (n) DETACH DELETE n")
        for label, prop in [("Vendor", "key"), ("RuleGroup", "id"), ("EntityRule", "id"), ("TableRule", "id"),
                            ("Document", "doc_id"), ("Page", "id"), ("Field", "id"), ("Table", "id"),
                            ("Entity", "key"), ("NormalizationRule", "name"), ("Invariant", "name")]:
            self.cypher(f"CREATE CONSTRAINT rm_{label.lower()}_{prop} IF NOT EXISTS "
                        f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")
        self.cypher("UNWIND $rules AS r MERGE (n:NormalizationRule {name: r.name}) SET n.rule = r.rule",
                    rules=[{"name": k, "rule": v} for k, v in H.NORMALIZATION_RULES.items()])
        self.cypher("MERGE (i:Invariant {name: 'arithmetic_check'}) SET i.expression = $expr",
                    expr=H.ARITHMETIC_INVARIANT)
        self.cypher("UNWIND $keys AS k MERGE (e:Entity {key: k.key}) SET e.kind = k.kind",
                    keys=[{"key": k, "kind": "table" if k in H.TABLE_TYPES else "entity"} for k in FEATURES])
        self.cypher("""MATCH (e:Entity) WHERE e.key IN keys($types) MATCH (r:NormalizationRule {name: $types[e.key]})
                       MERGE (e)-[:NORMALIZED_BY]->(r)""", types={k: H.VALUE_TYPES[k] for k in BOX_KEYS})
        self.cypher("""MATCH (e:Entity) WHERE e.key IN $members MATCH (i:Invariant {name: 'arithmetic_check'})
                       MERGE (e)-[:CONSTRAINED_BY]->(i)""",
                    members=sorted(H.INVARIANT_MEMBERS | {t for t, cols in H.TABLE_TYPES.items()
                                                          if set(cols) & H.INVARIANT_MEMBERS}))
        self.cypher("MERGE (v:Vendor {key: $key}) SET v.names = ['Unidentified']", key=UNIDENTIFIED)

        self._order = 0
        for doc_id in sorted(corpus.train_ids):
            rec = corpus.records[doc_id]
            for page in rec["pages"]:
                self.ingest_page(rec, page)
        self.rule_prior = self.estimate_rule_prior()
        for r in self.cypher("MATCH (g:RuleGroup) RETURN g.id AS id"):
            self.consolidate_group(r["id"])
        self._layouts = None
        self._check(corpus)

    def ingest_page(self, rec: dict, page: dict) -> dict:
        """Write one training page and place it: join the most similar rule group whose ranges hold it, or found
        a new one. Returns the decision (the ingestion log row)."""
        vendor = rec["vendor_key"] or UNIDENTIFIED
        path = page_path(page)
        pt, structure = page_type(page), self._structure(page)
        page_id = f"{rec['doc_id']}|p{page['number']}"
        if self.cypher("MATCH (p:Page {id: $id}) RETURN p", id=page_id):
            return {}                                                  # already ingested: idempotent
        if vendor != UNIDENTIFIED:
            self.cypher("MERGE (v:Vendor {key: $key}) ON CREATE SET v.names = [] "
                        "SET v.names = CASE WHEN $name IN v.names THEN v.names ELSE v.names + $name END",
                        key=vendor, name=rec["vendor_name"])
        self.cypher(WRITE_DOCUMENT, vendor=vendor, doc_id=rec["doc_id"], doc_type=H.DOC_TYPE)
        self.cypher(WRITE_PAGE, doc_id=rec["doc_id"], page_id=page_id, number=page["number"], page_type=pt,
                    signature=signature(path), structure=structure,
                    nodes=[{**n, "id": f"{page_id}|{n['key']}"} for n in path])

        checks = self.check_groups(vendor, pt, path, structure)
        passed = checks[checks.passed]
        if not passed.empty:
            best = passed.sort_values(["similarity", "n_pages"], ascending=False).iloc[0]
            group_id, role, similarity = best.group, "joined", round(float(best.similarity), 4)
            have = {r["key"] for r in self.cypher("MATCH (:RuleGroup {id: $id})-[:`_`*]->(r) RETURN r.key AS key",
                                                  id=group_id)}
        else:
            n = self.cypher("MATCH (:Vendor {key: $v})-[:HAS_RULE_GROUP]->(g:RuleGroup {page: $pt}) "
                            "RETURN count(g) AS n", v=vendor, pt=pt)[0]["n"]
            group_id, role, similarity, have = f"{vendor}|p{pt}|g{n}", "founded", None, set()
            self.cypher(FOUND_GROUP, vendor=vendor, group_id=group_id, doc_type=H.DOC_TYPE, page_type=pt,
                        zeros=[0.0] * len(structure))
        added = [{"kind": n["kind"], "key": n["key"]} for n in path if n["key"] not in have]
        if added or role == "founded":
            self.cypher(ADD_RULES, group_id=group_id, nodes=added)
            keys = sorted(have | {n["key"] for n in added}, key=NODE_ORDER.get)
            self.cypher(UNCHAIN, group_id=group_id)
            self.cypher(RECHAIN, group_id=group_id, ids=[f"{group_id}|{k}" for k in keys], signature=" > ".join(keys))
        self._order += 1
        self.cypher(JOIN_GROUP, group_id=group_id, page_id=page_id, role=role, order=self._order,
                    candidates=len(checks), passed=len(passed), similarity=similarity, structure=structure)
        self.consolidate_group(group_id)
        self._layouts = None
        return {"doc_id": rec["doc_id"], "page": page["number"], "group": group_id, "role": role,
                "candidates": len(checks), "passed": len(passed), "similarity": similarity}

    def _structure(self, page: dict) -> list[float]:
        return H.structure_vector(page, self.cfg.position_grid, self.cfg.position_weight)

    def check_groups(self, vendor: str, pt: int, path: list[dict], structure: list[float]) -> pd.DataFrame:
        """The matching rule, read-only: the vendor's rule groups whose structure vector has cosine >=
        rule_group_similarity with the page's, and for each whether every node the two share lies inside the
        group's range for it (`passed`)."""
        cols = ["group", "n_pages", "similarity", "checked", "inside", "outside", "passed"]
        groups = pd.DataFrame([{**r, "similarity": H.cosine(r["structure_vector"], structure)}
                               for r in self.cypher(CANDIDATE_GROUPS, vendor=vendor, doc_type=H.DOC_TYPE,
                                                    page_type=pt)],
                              columns=["group", "n_pages", "structure_vector", "similarity"])
        groups = groups[groups.similarity >= self.cfg.rule_group_similarity].drop(columns="structure_vector")
        if groups.empty:
            return pd.DataFrame(columns=cols).astype({"passed": bool})
        checked = self.cypher_df(RANGE_CHECK, groups=groups.group.tolist(),
                                 nodes=[{"key": n["key"], "point": n["point"]} for n in path])
        df = groups.merge(checked, on="group", how="left") if not checked.empty else \
            groups.assign(checked=0, inside=0, outside=None)
        df[["checked", "inside"]] = df[["checked", "inside"]].fillna(0).astype(int)
        df["outside"] = df["outside"].map(lambda o: o if isinstance(o, list) else [])
        df["passed"] = df.inside == df.checked
        return df[cols]

    def match_page(self, page: dict, vendor_name: str) -> str | None:
        """The ingestion rule applied to a page that is not written: the group it would join, or None when it
        would found a new one. For comparison with the gate on noisy test pages."""
        vendor = H.vendor_key(vendor_name) or UNIDENTIFIED
        checks = self.check_groups(vendor, page_type(page), page_path(page), self._structure(page))
        passed = checks[checks.passed]
        return None if passed.empty else passed.sort_values(["similarity", "n_pages"], ascending=False).group.iloc[0]

    # --- Consolidation ----------------------------------------------------------------------------------------

    def consolidate_group(self, group_id: str) -> None:
        """Rebuild every rule node of one group from the observations that support it."""
        n_pages = self.cypher("MATCH (g:RuleGroup {id: $id}) RETURN g.n_pages AS n", id=group_id)[0]["n"]
        by_rule = defaultdict(list)
        for o in self.cypher(GROUP_OBSERVATIONS, group_id=group_id):
            by_rule[(o["rule"], o["kind"])].append(o)
        rules = []
        for (rule_id, kind), obs in by_rule.items():
            props = consolidate_points(obs, n_pages, self.cfg, self.rule_prior[:2])
            if kind == "TableRule":
                cols = Counter(c for o in obs for c in (o["filled_columns"] or []))
                props["mean_rows"] = round(float(np.mean([o["rows"] for o in obs])), 1)
                props["columns"] = [c for c in H.TABLE_TYPES[obs[0]["key"]] if cols[c] / len(obs) >= 0.5]
                props["region_median"], props["bbox_median"] = props["bbox_median"], None
            rules.append({"id": rule_id, "props": props})
        self.cypher(SET_RULES, rules=rules)

    def estimate_rule_prior(self) -> tuple[float, float, float, int]:
        """Empirical-Bayes Beta prior for reliability: (alpha, beta, p0, cohorts used). p0 is how often one
        training box of an entity lands within box_range_min of another box of the same entity, over pages of
        one (vendor, page type, structure) cohort. Cohorts are formed from the document paths, not the groups,
        so pages the range check sent to different groups still count against each other."""
        rows = self.cypher("""
            MATCH (v:Vendor)-[:HAS_DOCUMENT]->(:Document)-[:HAS_PAGE]->(p:Page)-[:`_`*]->(f:Field)
            WHERE f.point IS NOT NULL
            RETURN v.key AS vendor, p.page_type AS page, p.signature AS signature, f.key AS key,
                   collect(f.point) AS points""")
        m, rates = self.cfg.box_range_min, []
        for r in rows:
            pts = r["points"]
            if len(pts) >= 2:
                rates.append(np.mean([abs(a[0] - b[0]) <= m and abs(a[1] - b[1]) <= m
                                      for i, a in enumerate(pts) for j, b in enumerate(pts) if i != j]))
        p0 = float(np.mean(rates)) if rates else 0.5
        s = self.cfg.rule_prior_strength
        return s * p0, s * (1 - p0), p0, len(rates)

    def _check(self, corpus: Corpus) -> None:
        assert self.train_doc_ids() == corpus.train_ids, "a training document is missing or duplicated"
        n_pages = sum(len(corpus.records[d]["pages"]) for d in corpus.train_ids)
        placed = self.cypher("MATCH (p:Page) OPTIONAL MATCH (p)-[m:IN_GROUP]->() RETURN p.id AS id, count(m) AS n")
        assert len(placed) == n_pages and all(r["n"] == 1 for r in placed), "a page is not in exactly one group"
        bad = self.cypher("""MATCH (p:Page)-[:IN_GROUP]->(g:RuleGroup), (p)-[:`_`*]->(f)
                             WHERE NOT EXISTS { (f)-[:SUPPORTS]->()<-[:`_`*]-(g) } RETURN f.id AS id""")
        assert not bad, f"a page node supports no rule of its group: {bad[:3]}"
        bad = self.cypher("""MATCH (g:RuleGroup)-[:`_`*]->(r) WHERE NOT EXISTS { (r)<-[:SUPPORTS]-() }
                             RETURN r.id AS id""")
        assert not bad, f"a rule with no supporting page: {bad[:3]}"
        bad = self.cypher("""MATCH (g:RuleGroup) WHERE size([(g)-[:`_`*]->(r) | r]) <> size(split(g.signature, ' > '))
                             AND g.signature <> '' RETURN g.id AS id""")
        assert not bad, f"a group path out of step with its signature: {bad[:3]}"
        counts = self.cypher("""MATCH (g:RuleGroup) OPTIONAL MATCH (g)<-[m:IN_GROUP {role: 'founded'}]-()
                                RETURN g.id AS id, count(m) AS founded, g.n_pages AS n""")
        assert all(r["founded"] == 1 for r in counts), "a group without exactly one founding page"
        assert sum(r["n"] for r in counts) == n_pages

    # --- Reading ----------------------------------------------------------------------------------------------

    @property
    def p_new(self) -> float:
        """Good–Turing mass of an unseen group: groups founded by a single training page ÷ training pages. The
        observed chance that the next page founds a new group."""
        r = self.cypher("MATCH (g:RuleGroup) RETURN sum(CASE WHEN g.n_pages = 1 THEN 1 ELSE 0 END) AS n1, "
                        "sum(g.n_pages) AS n")[0]
        return r["n1"] / r["n"]

    def layouts(self) -> dict[str, dict]:
        """Every rule group, plus the corpus-wide view per page type, in the shape the posterior reads:
        {id: {id, vendor, page, docs, n_filled: array over FEATURES, rules: {key: {range, q, bbox_median,
        strength}}, labels: {key: top_labels}}}. `q` is the rule's `reliability`; n_filled counts the group's
        pages that have the node."""
        if self._layouts is None:
            layouts = {}
            for r in self.cypher(GROUP_STATS):
                L = layouts.setdefault(r["group"], {"id": r["group"], "vendor": r["vendor"], "page": r["page"],
                                                    "docs": r["n_pages"], "n_filled": {}, "rules": {},
                                                    "labels": {}})
                if r["key"] is None:
                    continue
                L["n_filled"][r["key"]] = r["n_supported"]
                L["labels"][r["key"]] = r["top_labels"] or []
                if r["kind"] == "EntityRule" and r["range"]:
                    L["rules"][r["key"]] = {"range": r["range"], "q": r["reliability"],
                                            "bbox_median": r["bbox_median"], "strength": r["strength"]}
            corpus = {}
            for r in self.cypher(CORPUS_STATS):
                C = corpus.setdefault(r["page"], {"id": f"{ALL_VENDORS}|p{r['page']}", "vendor": ALL_VENDORS,
                                                  "page": r["page"], "docs": r["pages"], "n_filled": {},
                                                  "rules": {}, "labels": {}})
                if r["key"] is not None:
                    C["n_filled"][r["key"]] = r["n_filled"]
                    C["labels"][r["key"]] = [label for label, _ in Counter(r["labels"]).most_common(3)]
            layouts.update({C["id"]: C for C in corpus.values()})
            for L in layouts.values():
                L["n_filled"] = np.array([L["n_filled"].get(f, 0) for f in FEATURES], dtype=float)
            self._layouts = layouts
        return self._layouts

    def ingestion_log(self, vendor: str | None = None) -> pd.DataFrame:
        """Where every training page went, in ingestion order: the group, whether the page founded or joined it,
        how many of the vendor's groups were similar enough to be candidates and passed the range check, and the
        cosine with the group it joined."""
        return self.cypher_df("""
            MATCH (v:Vendor)-[:HAS_DOCUMENT]->(d:Document)-[:HAS_PAGE]->(p:Page)-[m:IN_GROUP]->(g:RuleGroup)
            WHERE $vendor IS NULL OR v.key = $vendor
            RETURN m.order AS order, v.key AS vendor, d.doc_id AS doc_id, p.number AS page, g.id AS group,
                   m.role AS role, m.candidates AS candidates, m.passed AS passed, m.similarity AS similarity,
                   size([(p)-[:`_`*]->(f) | f]) AS nodes
            ORDER BY order""", vendor=vendor)

    def groups_table(self) -> pd.DataFrame:
        """One row per rule group: vendor, page type, pages, nodes on its path, and its structure."""
        return self.cypher_df("""
            MATCH (v:Vendor)-[:HAS_RULE_GROUP]->(g:RuleGroup)
            RETURN v.key AS vendor, g.id AS group, g.document_type AS document_type, g.page AS page,
                   g.n_pages AS pages, size([(g)-[:`_`*]->(r) | r]) AS nodes, g.signature AS signature
            ORDER BY pages DESC, group""")

    def rules_table(self) -> pd.DataFrame:
        """One row per entity rule with a range: the consolidated statistics."""
        return self.cypher_df("""
            MATCH (v:Vendor)-[:HAS_RULE_GROUP]->(g:RuleGroup)-[:`_`*]->(r:EntityRule)
            WHERE r.range IS NOT NULL
            RETURN v.key AS vendor, g.id AS group, g.n_pages AS pages, r.key AS entity, r.fill_rate AS fill_rate,
                   r.n_boxes AS n_boxes, r.support AS support, r.spread AS spread, r.loo_hits AS loo_hits,
                   r.reliability AS reliability, r.strength AS strength, r.range AS range,
                   r.top_labels AS top_labels""")

    def entity_observations(self, group_id: str, key: str) -> dict:
        """The observations supporting one rule node, and the rule they consolidate to."""
        rule = self.cypher("""MATCH (r {id: $id}) WHERE r:EntityRule OR r:TableRule
                              RETURN r.bbox_center AS center, coalesce(r.bbox_median, r.region_median) AS median,
                                     r.range AS range, r.support AS support, r.n_boxes AS n_boxes,
                                     r.loo_hits AS loo_hits, r.reliability AS reliability, r.strength AS strength,
                                     r.fill_rate AS fill_rate, r.top_labels AS top_labels""",
                           id=f"{group_id}|{key}")[0]
        obs = [o for o in self.cypher(GROUP_OBSERVATIONS, group_id=group_id) if o["key"] == key]
        rule["observations"] = pd.DataFrame([{"doc_id": o["doc_id"], "point": o["point"], "bbox": o["bbox"],
                                              "label": o["label"]} for o in obs])
        return rule

    def layout_rules(self, group_id: str) -> pd.DataFrame:
        """Every node of one group's path, in path order: entities with their rule, tables with their columns."""
        rules = self.cypher_df("""
            MATCH (v:Vendor)-[:HAS_RULE_GROUP]->(g:RuleGroup {id: $id})
            MATCH path = (g)-[:`_`*]->(r)
            RETURN v.key AS vendor, g.id AS layout_id, g.n_pages AS layout_docs,
                   CASE WHEN r:TableRule THEN 'table' ELSE 'entity' END AS kind, r.key AS name,
                   length(path) AS order, r.fill_rate AS fill_rate, r.support AS support, r.n_boxes AS n_boxes,
                   r.strength AS strength, coalesce(r.bbox_median, r.region_median) AS bbox,
                   r.reliability AS reliability,
                   CASE WHEN r:TableRule THEN r.columns ELSE r.top_labels END AS top_labels
            ORDER BY order""", id=group_id)
        if not rules.empty:
            rules["top_labels"] = rules["top_labels"].map(lambda l: l if isinstance(l, list) else [])
        return rules

    def similar_layout_pairs(self, min_similarity: float) -> list[dict]:
        """Rule groups of different vendors on the same page type whose structure vectors (which nodes, and
        roughly where) have cosine >= min_similarity."""
        return self.cypher("""
            MATCH (va:Vendor)-[:HAS_RULE_GROUP]->(a:RuleGroup), (vb:Vendor)-[:HAS_RULE_GROUP]->(b:RuleGroup)
            WHERE va.key < vb.key AND a.page = b.page
            WITH va, vb, a, b, vector.similarity.cosine(a.structure_vector, b.structure_vector) AS sim
            WHERE sim >= $min
            RETURN va.key AS vendor_a, a.id AS layout_a, a.n_pages AS docs_a,
                   vb.key AS vendor_b, b.id AS layout_b, b.n_pages AS docs_b, a.page AS page, sim
            ORDER BY sim DESC""", min=min_similarity)
