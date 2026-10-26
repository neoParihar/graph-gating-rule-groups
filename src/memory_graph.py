"""The vendor-rooted memory graph in Neo4j: build (ingest + consolidate), read back, and the ungated (pass-1) guide.
The Cypher is gating.ipynb's, unchanged.

(:DocumentType)<-[:OF_TYPE]-(:Vendor)-[:HAS_LAYOUT]->(:PageLayout)-[:HAS_SECTION]->(:SectionLayout)
    -[:HAS_ENTITY]->(:EntityLayout)-[:NORMALIZED_BY|CONSTRAINED_BY]->(:NormalizationRule|:Invariant)
(:PageLayout)-[:HAS_TABLE]->(:TableLayout)-[:HAS_COLUMN]->(:ColumnLayout)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

import helpers as H
from src.config import Config
from src.data import Corpus

ALL_VENDORS = "ALL_VENDORS"
BOX_KEYS = [k for k in H.KV_KEYS if k not in H.DEFAULTED_KEYS]          # the 18 entities a rule can locate
FEATURES = BOX_KEYS + list(H.TABLE_TYPES)                                  # the 20 presence bits of a page

INGEST_PAGE = """
MERGE (v:Vendor {key: $vendor})
  ON CREATE SET v.names = [], v.train_doc_ids = []
WITH v, CASE WHEN $doc_id IN v.train_doc_ids THEN v.train_doc_ids ELSE v.train_doc_ids + $doc_id END AS doc_ids
SET v.names = CASE WHEN $vendor_name IN v.names OR $vendor_name = '' THEN v.names ELSE v.names + $vendor_name END,
    v.train_doc_ids = doc_ids, v.doc_count = size(doc_ids)
WITH v
MATCH (t:DocumentType {name: $doc_type})
MERGE (v)-[:OF_TYPE]->(t)
MERGE (v)-[:HAS_LAYOUT]->(p:PageLayout {id: $layout_id})
  ON CREATE SET p.page_number = $page_number, p.variant = $variant, p.doc_ids = [], p.fill_sum = [],
                p.structure_sum = []
WITH p, NOT ($doc_id IN p.doc_ids) AS is_new
WITH p, is_new, CASE WHEN is_new THEN p.doc_ids + $doc_id ELSE p.doc_ids END AS doc_ids,
     CASE WHEN NOT is_new THEN p.fill_sum
          WHEN size(p.fill_sum) = 0 THEN $fill_vector
          ELSE [i IN range(0, size($fill_vector) - 1) | p.fill_sum[i] + $fill_vector[i]] END AS fill_sum,
     CASE WHEN NOT is_new THEN p.structure_sum
          WHEN size(p.structure_sum) = 0 THEN $structure_vector
          ELSE [i IN range(0, size($structure_vector) - 1) | p.structure_sum[i] + $structure_vector[i]] END
       AS structure_sum
SET p.doc_ids = doc_ids, p.doc_count = size(doc_ids), p.fill_sum = fill_sum,
    p.fill_vector = [x IN fill_sum | x / size(doc_ids)],
    p.structure_sum = structure_sum, p.structure_vector = [x IN structure_sum | x / size(doc_ids)]
WITH p, is_new
UNWIND $fields AS f
MERGE (p)-[:HAS_SECTION {json_position: f.section_position}]->(s:SectionLayout {id: $layout_id + '|' + f.section})
  ON CREATE SET s.name = f.section
MERGE (s)-[:HAS_ENTITY {json_position: f.key_position}]->(e:EntityLayout {id: $layout_id + '|' + f.key})
  ON CREATE SET e.key = f.key, e.obs_doc_ids = [], e.obs_filled = [], e.obs_x = [], e.obs_y = [], e.obs_w = [],
                e.obs_h = [], e.obs_labels = []
WITH e, f, is_new WHERE is_new
SET e.obs_doc_ids = e.obs_doc_ids + $doc_id, e.obs_filled = e.obs_filled + f.filled,
    e.obs_x = e.obs_x + f.x, e.obs_y = e.obs_y + f.y, e.obs_w = e.obs_w + f.w, e.obs_h = e.obs_h + f.h,
    e.obs_labels = e.obs_labels + f.label
"""

INGEST_TABLES = """
MATCH (p:PageLayout {id: $layout_id})
UNWIND $tables AS t
MERGE (p)-[:HAS_TABLE]->(tl:TableLayout {id: $layout_id + '|' + t.type})
  ON CREATE SET tl.type = t.type, tl.obs_doc_ids = [], tl.obs_rows = [], tl.obs_x = [], tl.obs_y = [],
                tl.obs_w = [], tl.obs_h = []
WITH tl, t, NOT ($doc_id IN tl.obs_doc_ids) AS is_new
SET tl.obs_doc_ids = CASE WHEN is_new THEN tl.obs_doc_ids + $doc_id ELSE tl.obs_doc_ids END,
    tl.obs_rows = CASE WHEN is_new THEN tl.obs_rows + t.rows ELSE tl.obs_rows END,
    tl.obs_x = CASE WHEN is_new THEN tl.obs_x + t.region[0] ELSE tl.obs_x END,
    tl.obs_y = CASE WHEN is_new THEN tl.obs_y + t.region[1] ELSE tl.obs_y END,
    tl.obs_w = CASE WHEN is_new THEN tl.obs_w + t.region[2] ELSE tl.obs_w END,
    tl.obs_h = CASE WHEN is_new THEN tl.obs_h + t.region[3] ELSE tl.obs_h END
WITH tl, t, is_new
UNWIND t.columns AS c
MERGE (tl)-[:HAS_COLUMN {json_position: c.position}]->(cl:ColumnLayout {id: tl.id + '|' + c.header})
  ON CREATE SET cl.header = c.header, cl.obs_filled = [], cl.obs_x = []
WITH cl, c, is_new WHERE is_new
SET cl.obs_filled = cl.obs_filled + c.filled, cl.obs_x = cl.obs_x + c.x
"""

CONSOLIDATE_ENTITIES = """
MATCH (p:PageLayout)-[:HAS_SECTION]->()-[:HAS_ENTITY]->(e:EntityLayout)
WITH p, e, [i IN range(0, size(e.obs_x) - 1) WHERE e.obs_x[i] >= 0 |
            {x: e.obs_x[i], y: e.obs_y[i], w: e.obs_w[i], h: e.obs_h[i],
             cx: e.obs_x[i] + e.obs_w[i] / 2, cy: e.obs_y[i] + e.obs_h[i] / 2}] AS boxes
// 1. density peak: the box centre with the most neighbours within the tolerance
WITH p, e, boxes, [b IN boxes | size([o IN boxes WHERE abs(o.cx - b.cx) <= $tol AND abs(o.cy - b.cy) <= $tol])]
     AS density
WITH p, e, boxes, density, reduce(best = -1, i IN range(0, size(boxes) - 1) |
     CASE WHEN best = -1 OR density[i] > density[best] THEN i ELSE best END) AS peak
WITH p, e, boxes, CASE WHEN peak >= 0 THEN boxes[peak] END AS centre, CASE WHEN peak >= 0 THEN density[peak]
     ELSE 0 END AS support
WITH p, e, boxes, centre, support,
     [b IN boxes WHERE centre IS NOT NULL AND abs(b.cx - centre.cx) <= $tol AND abs(b.cy - centre.cy) <= $tol]
     AS members
// 2. medians of the supporting boxes, and the spread of all boxes around the peak
CALL (members, boxes, centre) {
  UNWIND (CASE WHEN size(members) = 0 THEN [null] ELSE members END) AS m
  WITH boxes, centre, percentileCont(m.x, 0.5) AS mx, percentileCont(m.y, 0.5) AS my,
       percentileCont(m.w, 0.5) AS mw, percentileCont(m.h, 0.5) AS mh
  UNWIND (CASE WHEN size(boxes) = 0 THEN [null] ELSE boxes END) AS b
  RETURN mx, my, mw, mh, percentileCont(sqrt((b.cx - centre.cx)^2 + (b.cy - centre.cy)^2), 0.5) AS spread
}
WITH p, e, boxes, centre, support, mx, my, mw, mh, spread,
     size(e.obs_filled) AS n_obs, size([f IN e.obs_filled WHERE f]) AS n_filled
// 3. labels
CALL (e) {
  UNWIND (CASE WHEN size([l IN e.obs_labels WHERE l <> '']) = 0 THEN [null] ELSE
          [l IN e.obs_labels WHERE l <> ''] END) AS label
  WITH label, count(label) AS n ORDER BY n DESC
  RETURN [l IN collect(label) WHERE l IS NOT NULL][0..3] AS top_labels
}
WITH e, p, n_obs, n_filled, size(boxes) AS n_boxes, support, centre, mx, my, mw, mh, spread, top_labels,
     CASE WHEN size(boxes) = 0 THEN 0.0 ELSE toFloat(support) / size(boxes) END AS support_fraction
WITH e, p, n_obs, n_filled, n_boxes, support, centre, mx, my, mw, mh, spread, top_labels, support_fraction,
     support_fraction >= $min_fraction AND support >= $min_docs AS consolidated
SET e.n_obs = n_obs, e.n_filled = n_filled, e.n_boxes = n_boxes,
    e.fill_rate = round(toFloat(n_filled) / n_obs, 3),
    e.support = support, e.support_fraction = round(support_fraction, 3),
    e.bbox_center = CASE WHEN centre IS NULL THEN null ELSE [round(centre.cx, 4), round(centre.cy, 4)] END,
    e.bbox_median = CASE WHEN centre IS NULL THEN null ELSE [round(mx, 4), round(my, 4), round(mw, 4), round(mh, 4)] END,
    e.bbox_spread = CASE WHEN spread IS NULL THEN null ELSE round(spread, 4) END,
    e.top_labels = top_labels,
    e.consolidated = consolidated,
    e.strength = CASE WHEN consolidated AND support >= $strong_docs THEN 'strong'
                      WHEN consolidated THEN 'weak'
                      WHEN p.doc_count = 1 AND centre IS NOT NULL THEN 'single'
                      ELSE 'none' END
"""

# 4. the approximate range around the peak, how many training centres it holds, and the Beta-Binomial
#    posterior predictive probability that the next centre falls inside it (the peak itself is not counted)
CONSOLIDATE_RANGES = """
MATCH (e:EntityLayout)
WITH e, CASE WHEN e.bbox_center IS NULL THEN null
             WHEN $k * e.bbox_spread < $min_margin THEN $min_margin
             WHEN $k * e.bbox_spread > $max_margin THEN $max_margin
             ELSE $k * e.bbox_spread END AS m
WITH e, CASE WHEN m IS NULL THEN null ELSE
     [CASE WHEN e.bbox_center[0] - m < 0 THEN 0.0 ELSE e.bbox_center[0] - m END,
      CASE WHEN e.bbox_center[1] - m < 0 THEN 0.0 ELSE e.bbox_center[1] - m END,
      CASE WHEN e.bbox_center[0] + m > 1 THEN 1.0 ELSE e.bbox_center[0] + m END,
      CASE WHEN e.bbox_center[1] + m > 1 THEN 1.0 ELSE e.bbox_center[1] + m END] END AS r
WITH e, r, CASE WHEN r IS NULL THEN 0 ELSE size([i IN range(0, size(e.obs_x) - 1) WHERE e.obs_x[i] >= 0
          AND r[0] <= e.obs_x[i] + e.obs_w[i] / 2 AND e.obs_x[i] + e.obs_w[i] / 2 <= r[2]
          AND r[1] <= e.obs_y[i] + e.obs_h[i] / 2 AND e.obs_y[i] + e.obs_h[i] / 2 <= r[3]]) END AS n_in
SET e.bbox_range = CASE WHEN r IS NULL THEN null ELSE [x IN r | round(x, 4)] END, e.n_in_range = n_in,
    e.p_in_range = CASE WHEN r IS NULL THEN null                 // the peak defines the range: not a trial
                        ELSE (n_in - 1 + $alpha) / (e.n_boxes - 1 + $alpha + $beta) END
"""

ORDER_ENTITIES = """
MATCH (p:PageLayout)-[:HAS_SECTION]->()-[:HAS_ENTITY]->(e:EntityLayout)
WITH p, e ORDER BY CASE WHEN e.bbox_center IS NULL THEN 2.0 ELSE e.bbox_center[1] END,
                   CASE WHEN e.bbox_center IS NULL THEN 2.0 ELSE e.bbox_center[0] END
WITH p, collect(e) AS entities
UNWIND range(0, size(entities) - 1) AS i
WITH entities[i] AS e, i
SET e.order = CASE WHEN e.bbox_center IS NULL THEN null ELSE i + 1 END
"""

CONSOLIDATE_TABLES = """
MATCH (p:PageLayout)-[:HAS_TABLE]->(t:TableLayout)
WITH p, t, [i IN range(0, size(t.obs_x) - 1) WHERE t.obs_x[i] >= 0 | i] AS idx
CALL (t, idx) {
  UNWIND (CASE WHEN size(idx) = 0 THEN [null] ELSE idx END) AS i
  RETURN percentileCont(t.obs_x[i], 0.5) AS x, percentileCont(t.obs_y[i], 0.5) AS y,
         percentileCont(t.obs_w[i], 0.5) AS w, percentileCont(t.obs_h[i], 0.5) AS h
}
SET t.presence_rate = round(toFloat(size(t.obs_doc_ids)) / p.doc_count, 3),
    t.mean_rows = round(toFloat(reduce(s = 0, r IN t.obs_rows | s + r)) / size(t.obs_rows), 1),
    t.region_bbox = CASE WHEN x IS NULL THEN null ELSE [round(x, 4), round(y, 4), round(w, 4), round(h, 4)] END
WITH t
MATCH (t)-[:HAS_COLUMN]->(c:ColumnLayout)
WITH c, [x IN c.obs_x WHERE x >= 0] AS xs
CALL (xs) {
  UNWIND (CASE WHEN size(xs) = 0 THEN [null] ELSE xs END) AS x
  RETURN percentileCont(x, 0.1) AS lo, percentileCont(x, 0.9) AS hi
}
SET c.fill_rate = round(toFloat(size([f IN c.obs_filled WHERE f])) / size(c.obs_filled), 3),
    c.x_band = CASE WHEN lo IS NULL THEN null ELSE [round(lo, 4), round(hi, 4)] END
"""

ENTITY_STATS = """
MATCH (v:Vendor)-[:HAS_LAYOUT]->(p:PageLayout)-[:HAS_SECTION]->()-[:HAS_ENTITY]->(e:EntityLayout)
RETURN v.key AS vendor, p.id AS layout_id, p.page_number AS page, p.doc_count AS docs, e.key AS key,
       e.n_filled AS n_filled, e.bbox_range AS bbox_range, e.p_in_range AS q, e.bbox_median AS bbox_median,
       e.strength AS strength, e.top_labels AS top_labels
"""

TABLE_STATS = """
MATCH (p:PageLayout)-[:HAS_TABLE]->(t:TableLayout)
RETURN p.id AS layout_id, t.type AS key, size(t.obs_doc_ids) AS n_filled
"""

NEAREST_LAYOUTS = """
MATCH (v:Vendor)-[:HAS_LAYOUT]->(p:PageLayout {page_number: $page})
WHERE v.key <> $all AND ($vendor IS NULL OR v.key = $vendor)
RETURN p.id AS layout_id, v.key AS vendor, p.doc_count AS docs,
       vector.similarity.cosine(p.structure_vector, $structure) AS structure_similarity
ORDER BY structure_similarity DESC, docs DESC LIMIT $k
"""


def assign_variants(records: dict, doc_ids: set[str], threshold: float) -> dict[tuple[str, int], int]:
    """(doc_id, page number) -> variant index within (vendor, page number). Greedy and deterministic: pages are
    visited in doc_id order and join the first variant whose running mean fill vector has cosine >= threshold with
    theirs, otherwise they open a new variant."""
    groups = defaultdict(list)                     # (vendor, page) -> [(centroid sum, count)]
    variant = {}
    for doc_id in sorted(doc_ids):
        rec = records[doc_id]
        for page in rec["pages"]:
            vec = np.array(H.fill_vector(page))
            slots = groups[(rec["vendor_key"], page["number"])]
            best = max(range(len(slots)), key=lambda i: H.cosine(slots[i][0] / slots[i][1], vec), default=None)
            if best is not None and H.cosine(slots[best][0] / slots[best][1], vec) >= threshold:
                slots[best] = (slots[best][0] + vec, slots[best][1] + 1)
            else:
                slots.append((vec.copy(), 1))
                best = len(slots) - 1
            variant[(doc_id, page["number"])] = best
    return variant


def table_payload(page: dict) -> list[dict]:
    """Per table type on the page: row count, the region its cell boxes cover, and per column the filled flags
    and x-centres of its cells (-1 when a cell has no box)."""
    payload = []
    for table in page["tables"]:
        columns = H.TABLE_TYPES[table["type"]]
        boxes = [c["bbox"] for row in table["rows"] for c in row.values() if c["bbox"]]
        if boxes:
            x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
            region = [x0, y0, x1 - x0, y1 - y0]
        else:
            region = [-1.0, -1.0, -1.0, -1.0]
        payload.append({"type": table["type"], "rows": len(table["rows"]), "region": region, "columns": [
            {"header": col, "position": i + 1,
             "filled": [bool(row.get(col, {}).get("value")) for row in table["rows"]],
             "x": [H.bbox_center(row[col]["bbox"])[0] if row.get(col, {}).get("bbox") else -1.0
                   for row in table["rows"]]}
            for i, col in enumerate(columns)]})
    return payload


class MemoryGraph:
    """One Neo4j database holding the memory. `build` writes it from a corpus's training documents; everything
    else reads it."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password),
                                           notifications_min_severity="OFF")
        self.driver.verify_connectivity()
        self.rule_prior: tuple[float, float, float, int] | None = None   # (alpha, beta, p0, layouts used)
        self._layouts: dict[str, dict] | None = None

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
        rows = self.cypher("MATCH (v:Vendor) WHERE v.key <> $all UNWIND v.train_doc_ids AS d RETURN d",
                           all=ALL_VENDORS)
        return {r["d"] for r in rows}

    # --- Build ------------------------------------------------------------------------------------------------

    def build(self, corpus: Corpus, reset: bool = True) -> None:
        """Ingest every training page into its vendor's tree and into ALL_VENDORS, link the special nodes, then
        consolidate. With reset=False an already-built graph for the same split is kept as it is."""
        if not reset and self.train_doc_ids() == corpus.train_ids:
            self._consolidate_prior_only()
            return
        self.cypher("MATCH (n) DETACH DELETE n")
        for label, prop in [("Vendor", "key"), ("PageLayout", "id"), ("SectionLayout", "id"),
                            ("EntityLayout", "id"), ("TableLayout", "id"), ("ColumnLayout", "id"),
                            ("NormalizationRule", "name"), ("Invariant", "name"), ("DocumentType", "name")]:
            self.cypher(f"CREATE CONSTRAINT {label.lower()}_{prop} IF NOT EXISTS "
                        f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")
        self.cypher("MERGE (:DocumentType {name: $name})", name=H.DOC_TYPE)
        self.cypher("UNWIND $rules AS r MERGE (n:NormalizationRule {name: r.name}) SET n.rule = r.rule",
                    rules=[{"name": k, "rule": v} for k, v in H.NORMALIZATION_RULES.items()])
        self.cypher("MERGE (i:Invariant {name: 'arithmetic_check'}) SET i.expression = $expr",
                    expr=H.ARITHMETIC_INVARIANT)

        variants = assign_variants(corpus.records, corpus.train_ids, self.cfg.structure_similarity)
        for doc_id in sorted(corpus.train_ids):
            rec = corpus.records[doc_id]
            for page in rec["pages"]:
                self._ingest_page(rec, page, rec["vendor_key"], variants[(doc_id, page["number"])])
                self._ingest_page(rec, page, ALL_VENDORS, 0)
        self.cypher("""MATCH (e:EntityLayout) MATCH (r:NormalizationRule {name: $types[e.key]})
                       MERGE (e)-[:NORMALIZED_BY]->(r)""", types=H.VALUE_TYPES)
        self.cypher("""MATCH (n) WHERE (n:EntityLayout AND n.key IN $members) OR (n:ColumnLayout AND n.header IN $members)
                       MATCH (i:Invariant {name: 'arithmetic_check'}) MERGE (n)-[:CONSTRAINED_BY]->(i)""",
                    members=sorted(H.INVARIANT_MEMBERS))

        assert self.train_doc_ids() == corpus.train_ids, "a training document is missing or duplicated"
        bad = self.cypher("""MATCH (p:PageLayout)-[:HAS_SECTION]->()-[:HAS_ENTITY]->(e)
                             WHERE size(e.obs_doc_ids) <> p.doc_count OR size(e.obs_x) <> p.doc_count
                             RETURN e.id AS id""")
        assert not bad, bad[:3]
        self.consolidate()

    def _ingest_page(self, rec: dict, page: dict, vendor: str, variant: int) -> None:
        layout_id = f"{vendor}|p{page['number']}|v{variant}"
        section_position = {name: i + 1 for i, name in enumerate(H.SECTIONS)}
        fields = []
        for key in H.KV_KEYS:
            f = page["fields"].get(key, {"value": "", "bbox": None, "label": ""})
            x, y, w, h = f["bbox"] or [-1.0, -1.0, -1.0, -1.0]
            section = H.SECTION_OF[key]
            fields.append({"key": key, "section": section, "section_position": section_position[section],
                           "key_position": H.SECTIONS[section].index(key) + 1, "filled": bool(f["value"]),
                           "x": x, "y": y, "w": w, "h": h, "label": f["label"]})
        self.cypher(INGEST_PAGE, vendor=vendor, vendor_name=rec["vendor_name"] if vendor != ALL_VENDORS else "",
                    doc_id=rec["doc_id"], doc_type=H.DOC_TYPE, layout_id=layout_id, page_number=page["number"],
                    variant=variant, fill_vector=H.fill_vector(page),
                    structure_vector=H.structure_vector(page, self.cfg.position_grid, self.cfg.position_weight),
                    fields=fields)
        # "General" has no entities but is still part of every page's path
        self.cypher("""MATCH (p:PageLayout {id: $layout_id})
                       MERGE (p)-[:HAS_SECTION {json_position: $pos}]->(s:SectionLayout {id: $layout_id + '|General'})
                       SET s.name = 'General'""", layout_id=layout_id, pos=section_position["General"])
        self.cypher(INGEST_TABLES, layout_id=layout_id, doc_id=rec["doc_id"], tables=table_payload(page))

    # --- Consolidation ----------------------------------------------------------------------------------------

    def estimate_rule_prior(self) -> tuple[float, float, float, int]:
        """Empirical-Bayes Beta prior for p_in_range: (alpha, beta, p0, layouts used). p0 is the out-of-sample hit
        rate of a one-document rule: over ordered pairs of boxes of one vendor layout, how often centre j lies
        within box_range_min of centre i. Averaged per layout, then over layouts."""
        rates = []
        m = self.cfg.box_range_min
        for r in self.cypher("""MATCH (v:Vendor)-[:HAS_LAYOUT]->()-[:HAS_SECTION]->()-[:HAS_ENTITY]->(e:EntityLayout)
                                WHERE v.key <> $all AND NOT e.key IN $defaulted
                                RETURN e.obs_x AS x, e.obs_y AS y, e.obs_w AS w, e.obs_h AS h""",
                             all=ALL_VENDORS, defaulted=sorted(H.DEFAULTED_KEYS)):
            centres = [(x + w / 2, y + h / 2) for x, y, w, h in zip(r["x"], r["y"], r["w"], r["h"]) if x >= 0]
            if len(centres) >= 2:
                rates.append(np.mean([abs(a[0] - b[0]) <= m and abs(a[1] - b[1]) <= m
                                      for i, a in enumerate(centres) for j, b in enumerate(centres) if i != j]))
        p0 = float(np.mean(rates))
        s = self.cfg.rule_prior_strength
        return s * p0, s * (1 - p0), p0, len(rates)

    def consolidate(self) -> None:
        cfg = self.cfg
        self.rule_prior = self.estimate_rule_prior()
        self.cypher(CONSOLIDATE_ENTITIES, tol=cfg.bbox_tolerance, min_fraction=cfg.min_support_fraction,
                    min_docs=cfg.min_support_docs, strong_docs=cfg.strong_support_docs)
        self.cypher(CONSOLIDATE_RANGES, k=cfg.box_range_spread_k, min_margin=cfg.box_range_min,
                    max_margin=cfg.box_range_max, alpha=self.rule_prior[0], beta=self.rule_prior[1])
        self.cypher(ORDER_ENTITIES)
        self.cypher(CONSOLIDATE_TABLES)
        self._layouts = None

    def _consolidate_prior_only(self) -> None:
        self.rule_prior = self.estimate_rule_prior()
        self._layouts = None

    # --- Reading ----------------------------------------------------------------------------------------------

    def layouts(self) -> dict[str, dict]:
        """Every PageLayout's counts and rules, pulled out of the graph once (the posterior runs in numpy):
        {layout_id: {id, vendor, page, docs, n_filled: array over FEATURES, rules: {key: {range, q, bbox_median,
        strength}}, labels: {key: top_labels}}}"""
        if self._layouts is None:
            layouts = {}
            for r in self.cypher(ENTITY_STATS):
                L = layouts.setdefault(r["layout_id"], {"id": r["layout_id"], "vendor": r["vendor"],
                                                        "page": r["page"], "docs": r["docs"], "n_filled": {},
                                                        "rules": {}, "labels": {}})
                L["n_filled"][r["key"]] = r["n_filled"]
                L["labels"][r["key"]] = r["top_labels"]
                if r["bbox_range"] and r["key"] in BOX_KEYS:
                    L["rules"][r["key"]] = {"range": r["bbox_range"], "q": r["q"], "bbox_median": r["bbox_median"],
                                            "strength": r["strength"]}
            for r in self.cypher(TABLE_STATS):
                layouts[r["layout_id"]]["n_filled"][r["key"]] = r["n_filled"]
            for L in layouts.values():
                L["n_filled"] = np.array([L["n_filled"].get(f, 0) for f in FEATURES], dtype=float)
            self._layouts = layouts
        return self._layouts

    def rules_table(self) -> pd.DataFrame:
        """One row per vendor-tree entity rule with a range: the consolidated statistics."""
        return self.cypher_df("""
            MATCH (v:Vendor)-[:HAS_LAYOUT]->(p:PageLayout)-[:HAS_SECTION]->()-[:HAS_ENTITY]->(e:EntityLayout)
            WHERE v.key <> $all AND NOT e.key IN $defaulted AND e.bbox_range IS NOT NULL
            RETURN v.key AS vendor, p.id AS layout, p.doc_count AS docs, e.key AS entity, e.n_boxes AS n_boxes,
                   e.support AS support, e.n_in_range AS n_in_range, e.p_in_range AS p_in_range,
                   e.strength AS strength, e.bbox_range AS bbox_range, e.top_labels AS top_labels""",
                              all=ALL_VENDORS, defaulted=sorted(H.DEFAULTED_KEYS))

    def entity_observations(self, layout_id: str, key: str) -> dict:
        """The raw observations and consolidated rule of one EntityLayout."""
        return self.cypher("""MATCH (e:EntityLayout {id: $id})
                              RETURN e.obs_doc_ids AS doc_ids, e.obs_x AS x, e.obs_y AS y, e.obs_w AS w, e.obs_h AS h,
                                     e.bbox_center AS center, e.bbox_median AS median, e.bbox_range AS range,
                                     e.support AS support, e.n_boxes AS n_boxes, e.n_in_range AS n_in_range,
                                     e.p_in_range AS q, e.strength AS strength""", id=f"{layout_id}|{key}")[0]

    def layout_rules(self, layout_id: str) -> pd.DataFrame:
        """Every rule-bearing entity path and table path of one PageLayout, in JSON order."""
        entities = self.cypher_df("""
            MATCH (v:Vendor)-[:HAS_LAYOUT]->(p:PageLayout {id: $id})-[hs:HAS_SECTION]->(s)-[he:HAS_ENTITY]->(e)
            WHERE (e.strength <> 'none' OR size(e.top_labels) > 0) AND NOT e.key IN $defaulted
            RETURN v.key AS vendor, p.id AS layout_id, p.doc_count AS layout_docs, 'entity' AS kind,
                   s.name AS section, e.key AS name, hs.json_position AS sec_pos, he.json_position AS pos,
                   e.order AS order, e.fill_rate AS fill_rate, e.support AS support, e.n_boxes AS n_boxes,
                   e.strength AS strength, e.bbox_median AS bbox, e.top_labels AS top_labels
            ORDER BY sec_pos, pos""", id=layout_id, defaulted=sorted(H.DEFAULTED_KEYS))
        tables = self.cypher_df("""
            MATCH (v:Vendor)-[:HAS_LAYOUT]->(p:PageLayout {id: $id})-[:HAS_TABLE]->(t)
            OPTIONAL MATCH (t)-[hc:HAS_COLUMN]->(c) WHERE c.fill_rate >= 0.5
            WITH v, p, t, c ORDER BY hc.json_position
            RETURN v.key AS vendor, p.id AS layout_id, p.doc_count AS layout_docs, 'table' AS kind,
                   'Tables' AS section, t.type AS name, 99 AS sec_pos, 0 AS pos, null AS order,
                   t.presence_rate AS fill_rate, size(t.obs_doc_ids) AS support, null AS n_boxes,
                   CASE WHEN t.presence_rate >= 0.5 THEN 'strong' ELSE 'weak' END AS strength,
                   t.region_bbox AS bbox, collect(c.header) AS top_labels""", id=layout_id)
        rules = pd.concat([entities, tables], ignore_index=True)
        if not rules.empty:   # keep list columns as lists (concat can turn missing values into NaN)
            rules["bbox"] = rules["bbox"].map(lambda b: b if isinstance(b, list) else None)
            rules["top_labels"] = rules["top_labels"].map(lambda l: l if isinstance(l, list) else [])
        return rules

    # --- The ungated (pass-1) guide ---------------------------------------------------------------------------

    def nearest_layouts(self, structure: list[float], page_number: int, vendor: str | None = None,
                        k: int = 5) -> list[dict]:
        """The k vendor PageLayouts with the highest structure-vector cosine (of one vendor, or of any)."""
        return self.cypher(NEAREST_LAYOUTS, page=min(page_number, 2), vendor=vendor, all=ALL_VENDORS,
                           structure=structure, k=k)

    def memory_guide(self, vendor_name: str, page: dict) -> dict:
        """The paths that guide one page in the ungated pass: the vendor's closest layout if the vendor is known,
        otherwise the guide_neighbours closest layouts of any vendor; always the ALL_VENDORS layout."""
        vkey = H.vendor_key(vendor_name)
        structure = H.structure_vector(page, self.cfg.position_grid, self.cfg.position_weight)
        own = self.nearest_layouts(structure, page["number"], vendor=vkey, k=1)
        if own:
            chosen = [{**own[0], "source": "same vendor"}]
        else:
            chosen = [{**n, "source": "similar structure, other vendor"}
                      for n in self.nearest_layouts(structure, page["number"], k=self.cfg.guide_neighbours)]
        chosen.append({"layout_id": f"{ALL_VENDORS}|p{min(page['number'], 2)}|v0", "vendor": ALL_VENDORS,
                       "source": "corpus labels", "structure_similarity": None})
        return {"vendor_key": vkey, "vendor_known": bool(own), "page": page["number"], "layouts": chosen}

    def rules_to_text(self, guide: dict) -> str:
        """Render an ungated guide as prompt text: one line per entity rule (box + label), one per table."""
        lines = [f"PAGE {guide['page']}"]
        for layout in guide["layouts"]:
            rules = self.layout_rules(layout["layout_id"])
            if rules.empty:
                continue
            sim = layout.get("structure_similarity")
            lines.append(f"  Layout (vendor tree '{layout['vendor']}', {layout['source']}"
                         + (f", structure similarity {sim:.2f}" if sim is not None else "")
                         + f", built from {rules.layout_docs.iloc[0]} invoice(s)):")
            corpus = layout["vendor"] == ALL_VENDORS
            for r in rules.itertuples():
                labels = f" labelled {', '.join(repr(l) for l in r.top_labels)}" if r.top_labels else ""
                if r.kind == "table":
                    where = f" in region box_2d {H.bbox_to_box2d(r.bbox)}" if r.bbox and not corpus else ""
                    lines.append(f"    TABLE {r.name}: present on {r.fill_rate:.0%} of pages{where}; usually "
                                 f"filled columns: {', '.join(r.top_labels) or '-'}")
                elif corpus:
                    if r.top_labels:
                        lines.append(f"    {r.name}: common labels {', '.join(repr(l) for l in r.top_labels)}")
                else:
                    where = (f"value usually at box_2d {H.bbox_to_box2d(r.bbox)} ({r.strength}: {r.support} of "
                             f"{r.n_boxes} located)" if r.bbox and r.strength != "none" else "position varies")
                    lines.append(f"    {r.name}: filled on {r.fill_rate:.0%} of pages; {where}{labels}")
        return "\n".join(lines)
