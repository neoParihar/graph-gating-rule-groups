"""What a page looks like to the memory: the entities and tables it can hold, and the page as a path of nodes."""

from __future__ import annotations

import helpers as H

BOX_KEYS = [k for k in H.KV_KEYS if k not in H.DEFAULTED_KEYS]        # the 18 entities a rule can locate
FEATURES = BOX_KEYS + list(H.TABLE_TYPES)                                # the 20 presence bits of a page
NODE_ORDER = {key: i for i, key in enumerate(FEATURES)}                  # path order: entities (JSON order), tables
CORPUS_VIEW = "ALL_VENDORS"                # vendor key of the corpus-wide view (the posterior's NEW behaves like it)
UNIDENTIFIED = H.UNCLASSIFIED              # vendor key of pages printed without a vendor name


def page_type(page: dict) -> int:
    """1 for a first page, 2 for every continuation page."""
    return min(page["number"], 2)


def table_regions(page: dict) -> list[dict]:
    """Per table on the page: its row count, the region its cell boxes cover ([x, y, w, h], or -1s when no cell has
    a box), and per column the filled flags and x-centres of its cells (-1 when a cell has no box)."""
    tables = []
    for table in page["tables"]:
        columns = H.TABLE_TYPES[table["type"]]
        boxes = [c["bbox"] for row in table["rows"] for c in row.values() if c["bbox"]]
        if boxes:
            x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
            region = [x0, y0, x1 - x0, y1 - y0]
        else:
            region = [-1.0, -1.0, -1.0, -1.0]
        tables.append({"type": table["type"], "rows": len(table["rows"]), "region": region, "columns": [
            {"header": col, "position": i + 1,
             "filled": [bool(row.get(col, {}).get("value")) for row in table["rows"]],
             "x": [H.bbox_center(row[col]["bbox"])[0] if row.get(col, {}).get("bbox") else -1.0
                   for row in table["rows"]]}
            for i, col in enumerate(columns)]})
    return tables


def page_path(page: dict) -> list[dict]:
    """The page as a path: one node per filled entity and per table with rows, in path order. `point` is what the
    range check tests: an entity's box centre, a table's top-left corner (a table grows downward with its rows).
    None when the value has no usable box."""
    nodes = []
    for key in BOX_KEYS:
        f = page["fields"].get(key)
        if f and f["value"]:
            nodes.append({"kind": "entity", "key": key, "bbox": f["bbox"], "label": f.get("label", ""),
                          "point": list(H.bbox_center(f["bbox"])) if f["bbox"] else None})
    for t in table_regions(page):
        if t["rows"]:
            region = t["region"] if t["region"][0] >= 0 else None
            nodes.append({"kind": "table", "key": t["type"], "bbox": region, "rows": t["rows"], "label": "",
                          "point": region[:2] if region else None,
                          "filled_columns": [c["header"] for c in t["columns"] if any(c["filled"])]})
    return sorted(nodes, key=lambda n: NODE_ORDER[n["key"]])


def path_signature(path: list[dict]) -> str:
    """The path's structure: its node kinds in order, e.g. 'Vendor Name > Invoice Number > Item Line Table'."""
    return " > ".join(n["key"] for n in path)


def page_presence(page: dict) -> list[float]:
    """1 for every filled entity and every table with rows, in FEATURES order: what the fields term compares."""
    filled = {k for k, f in page["fields"].items() if f["value"]} | {t["type"] for t in page["tables"] if t["rows"]}
    return [float(f in filled) for f in FEATURES]
