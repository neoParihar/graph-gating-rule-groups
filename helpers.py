"""Support code for gating.ipynb.

It covers the Invoice catalog constants, ground-truth JSON parsing, the one place the input PDFs are read, bbox
arithmetic, position bins and box ranges, small probability utilities, Gemini schemas, prompts and plumbing, and
string similarity. The notebook holds the configuration, connections, Cypher, consolidation, the posterior over
layouts, both guide tools and scoring.


###!FUTURE DATA REQUIREMENTS!###

The ground-truth JSON does not carry the items below, so this notebook reads them from the input PDFs (only in
helpers.pdf_page_facts). Future ground-truth OCR outputs should include them, so the memory can be built from the
JSON alone.

1. Page dimensions and units
   Needed for: turning value_bbox into page fractions, so that boxes from different page sizes, scans and Gemini's
               0-1000 grid are comparable.
   Current workaround: the page rectangle of the PDF (points). Scanned pages are thought to carry boxes in image
               pixels, so their fractions may be off. Boxes outside the page are marked unusable.
   Add to each page: "width", "height", "unit" ("pt" | "px").

2. Label (key) text and position
   Needed for: EntityLayout.top_labels, i.e. which printed label sits next to a value ("Invoice #", "Amount Due").
   Current workaround: key_bbox is always [[0,0,0,0]], so the label is the run of PDF words just left of or above
               value_bbox. Scans have no text layer, so they have no labels.
   Add to each value: a real key_bbox and "key_text" (the label exactly as printed).

3. Page source (born-digital or scan) and collection origin
   Needed for: explaining unit differences and missing labels, and grouping documents by where they came from.
   Current workaround: a page counts as a scan when the PDF has no text layer.
   Add to each document: "source" ("digital" | "scan") and where the PDF was collected.

4. OCR token stream
   Needed for: resolving word_index ([[page, token]]) into words, which would replace item 2's PDF lookup.
   Add to each document: the token list that word_index points into (text plus box per token).

"""

from __future__ import annotations

import difflib
import json
import math
import re
from pathlib import Path
from typing import Any

import pymupdf
from google.genai import errors, types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

# ---------------------------------------------------------------------------
# Invoice catalog 
# ---------------------------------------------------------------------------

DOC_TYPE = "Invoice"

# Section order and entity order exactly as they appear in every ground-truth page.
SECTIONS: dict[str, list[str]] = {
    "Primary Fields": ["Vendor Name", "Invoice Number", "PO Number", "Invoice Date", "Invoice Amount"],
    "Invoice Info": ["Invoice TAX Amount", "Terms", "Freight Amount", "Terms Currency", "Due Date"],
    "Vendor Data": ["Customer Account Number", "Postal Code", "VAT Number", "Phone", "Email", "TAX ID"],
    "Bank Details": ["Bank Account", "Routing Number", "Bank Account Number"],
    "General": [],
}
KV_KEYS = [key for keys in SECTIONS.values() for key in keys]
SECTION_OF = {key: section for section, keys in SECTIONS.items() for key in keys}

TABLE_TYPES: dict[str, list[str]] = {
    "Item Line Table": ["Item", "Item Quantity", "Item Unit of Measurement", "Item Memo",
                        "Unit Price", "Line Item Amount", "VAT Rate", "VAT Amount"],
    "Expense Table": ["Expense", "Expense Amount", "Expense Memo"],
}

VALUE_TYPES: dict[str, str] = {
    **{key: "TEXT" for key in KV_KEYS},
    "Invoice Date": "DATE", "Due Date": "DATE",
    "Invoice Amount": "AMOUNT", "Invoice TAX Amount": "AMOUNT", "Freight Amount": "AMOUNT",
    "Terms Currency": "CURRENCY",
    **{col: "TEXT" for cols in TABLE_TYPES.values() for col in cols},
    "Unit Price": "AMOUNT", "Line Item Amount": "AMOUNT", "VAT Amount": "AMOUNT", "Expense Amount": "AMOUNT",
    "Item Quantity": "NUMBER", "VAT Rate": "RATE",
}

NORMALIZATION_RULES: dict[str, str] = {
    "DATE": "MM/DD/YYYY, zero-padded: 7/14/2026 -> 07/14/2026",
    "AMOUNT": "Digits with exactly 2 decimals, no currency symbols or thousands separators, negative = leading '-': "
              "$1,545.59 -> 1545.59",
    "TEXT": "Verbatim as printed; multi-line values joined with single spaces",
    "NUMBER": "Number as printed; '1' when no quantity is printed",
    "RATE": "Rate as printed, with or without '%'",
    "CURRENCY": "ISO currency code; a default (USD), not read from the page",
}

ARITHMETIC_INVARIANT = "sum(Line Item Amount) + sum(VAT Amount) + sum(Expense Amount) = Invoice Amount"
INVARIANT_MEMBERS = {"Invoice Amount", "Line Item Amount", "VAT Amount", "Expense Amount"}

# Filled by default rather than read from the page: no location, so never learned or scored.
DEFAULTED_KEYS = {"Terms Currency"}

DESCRIPTIONS: dict[str, str] = {
    "Vendor Name": "Company issuing the invoice, usually in the letterhead.",
    "Invoice Number": "The invoice's identifier, as printed.",
    "PO Number": "Purchase order reference, verbatim from the P.O. field even if it is a name or budget label.",
    "Invoice Date": "Date the invoice was issued.",
    "Invoice Amount": "Invoice total / amount due.",
    "Invoice TAX Amount": "Total tax on the invoice.",
    "Terms": "Payment terms; may be a short code or a whole sentence.",
    "Freight Amount": "Freight / shipping total.",
    "Terms Currency": "Always USD.",
    "Due Date": "Date payment is due.",
    "Customer Account Number": "The buyer's account id at the vendor.",
    "Postal Code": "The vendor's ZIP / postal code, as printed.",
    "VAT Number": "The vendor's VAT registration number.",
    "Phone": "The vendor's phone number, original formatting kept.",
    "Email": "The vendor's email address.",
    "TAX ID": "The vendor's EIN / tax id, as printed.",
    "Bank Account": "The NAME of the vendor's bank, not a number.",
    "Routing Number": "ABA routing number, as printed.",
    "Bank Account Number": "The vendor's remit-to bank account number.",
}

# ---------------------------------------------------------------------------
# Data the ground-truth JSON does not carry, so it is read from the PDF for now
# ---------------------------------------------------------------------------

def pdf_page_facts(pdf_path: Path) -> dict[int, dict[str, Any]]:
    """The only place an input PDF is read: per page number, its size (points), whether it has a text layer, and
    its words (x0, y0, x1, y1, text) for label lookup. See FUTURE_DATA_REQUIREMENTS for why."""
    with pymupdf.open(pdf_path) as pdf:
        facts = {}
        for i, page in enumerate(pdf, start=1):
            words = [w[:5] for w in page.get_text("words")]
            facts[i] = {"width": page.rect.width, "height": page.rect.height,
                        "has_text_layer": bool(words), "words": words}
        return facts


# ---------------------------------------------------------------------------
# Small value utilities
# ---------------------------------------------------------------------------


def is_empty(value: Any) -> bool:
    return value is None or not str(value).strip()


def normalize_for_match(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def string_similarity(a: Any, b: Any) -> float:
    """0..1 character-level similarity of two values after whitespace/case normalization (difflib ratio)."""
    a, b = normalize_for_match(a), normalize_for_match(b)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


_VENDOR_SUFFIXES = {"inc", "llc", "ltd", "corp", "co", "corporation", "company", "com", "the"}
UNCLASSIFIED = "UNCLASSIFIED"


def vendor_key(name: Any) -> str:
    """Merge spelling variants of one vendor ('Solodev' / 'solodev', 'Zoro Tools, Inc.' / 'zoro tools').
    Documents without a Vendor Name get the UNCLASSIFIED root."""
    tokens = re.sub(r"[^a-z0-9 ]", " ", str(name or "").lower()).split()
    return " ".join(t for t in tokens if t not in _VENDOR_SUFFIXES) or UNCLASSIFIED


# ---------------------------------------------------------------------------
# Bounding boxes. Stored as [x, y, w, h] page fractions (0..1, top-left origin).
# Gemini reads and writes box_2d = [ymin, xmin, ymax, xmax] on a 0..1000 grid.
# ---------------------------------------------------------------------------


def normalize_bbox(bbox: list[float], width: float, height: float) -> list[float] | None:
    """Raw [x, y, w, h] in page units -> page fractions. None for empty boxes and boxes that leave the page."""
    x, y, w, h = (float(v) for v in bbox)
    if w <= 0 or h <= 0 or x + w > width * 1.02 or y + h > height * 1.02:
        return None
    return [round(min(max(v, 0.0), 1.0), 4) for v in (x / width, y / height, w / width, h / height)]


def denormalize_bbox(bbox: list[float] | None, width: float, height: float) -> list[float]:
    if not bbox:
        return [0.0, 0.0, 0.0, 0.0]
    x, y, w, h = bbox
    return [x * width, y * height, w * width, h * height]


def bbox_to_box2d(bbox: list[float]) -> list[int]:
    x, y, w, h = bbox
    return [round(y * 1000), round(x * 1000), round((y + h) * 1000), round((x + w) * 1000)]


def box2d_to_bbox(box: list[float] | None) -> list[float] | None:
    if not box or len(box) != 4:
        return None
    ymin, xmin, ymax, xmax = (min(max(float(v), 0.0), 1000.0) / 1000 for v in box)
    if xmax <= xmin or ymax <= ymin:
        return None
    return [xmin, ymin, xmax - xmin, ymax - ymin]


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def bbox_overlap(a: list[float] | None, b: list[float] | None) -> float:
    """Intersection over union of two [x, y, w, h] boxes; 0.0 when either is missing."""
    if not a or not b:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: list[float] | None, b: list[float] | None) -> float | None:
    """Euclidean distance between box centres in page fractions (0 = same spot, ~1.4 = opposite corners)."""
    if not a or not b:
        return None
    (ax, ay), (bx, by) = bbox_center(a), bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Label lookup: the printed words just left of (same line) or above a value, e.g. "Invoice #"
# ---------------------------------------------------------------------------


def _is_label_token(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text)) or text in {"#", "#:"}


def anchor_label(words: list[tuple], bbox: list[float], max_gap: float = 200, max_words: int = 5) -> str:
    """`words` are (x0, y0, x1, y1, text) in PDF points and `bbox` is a raw [x, y, w, h] value box."""
    x, y, w, h = bbox
    cy = y + h / 2

    def outside(wd: tuple) -> bool:
        return not (wd[0] >= x - 1 and wd[2] <= x + w + 1 and wd[1] >= y - 1 and wd[3] <= y + h + 1)

    labels = [wd for wd in words if _is_label_token(wd[4]) and outside(wd)]
    left = sorted((wd for wd in labels if wd[1] <= cy <= wd[3] and wd[2] <= x + 1 and x - wd[2] < max_gap),
                  key=lambda wd: wd[0])
    if left:
        run = [left[-1]]
        for wd in reversed(left[:-1]):
            if run[0][0] - wd[2] > 12:
                break
            run.insert(0, wd)
        return " ".join(wd[4] for wd in run[-max_words:]).rstrip(":").strip()
    above = [wd for wd in labels if wd[3] <= y + 1 and y - wd[3] < 40 and wd[0] < x + w and wd[2] > x]
    if above:
        bottom = max(wd[3] for wd in above)
        line = sorted((wd for wd in above if abs(wd[3] - bottom) < 3), key=lambda wd: wd[0])
        return " ".join(wd[4] for wd in line[:max_words]).rstrip(":").strip()
    return ""


# ---------------------------------------------------------------------------
# Ground-truth (or pass-output) JSON -> flat document record
# ---------------------------------------------------------------------------


def _grounded(word_index: Any) -> bool:
    """True when word_index points into the page's OCR tokens: [[page, token]] (not [], [[]] or [None])."""
    return bool(word_index) and isinstance(word_index[0], list) and len(word_index[0]) == 2


def flatten_document(doc_id: str, pages_json: list[dict], page_facts: dict[int, dict]) -> dict[str, Any]:
    """One output-schema document (ground truth, a saved Gemini pass, or a fabricated doc) -> a flat record:

    {doc_id, vendor_name, vendor_key, page_count, is_scanned,
     pages: [{number, fields: {key: {value, bbox, label, valid, usable}}, tables: [{type, rows: [{col: cell}]}]}]}

    `bbox` is in page fractions (None when empty or unusable). `usable` = filled, grounded or unknown, valid, and
    a box that fits the page. Only usable boxes are learned and scored."""
    pages = []
    for pg in pages_json:
        number = int(pg["number"])
        facts = page_facts.get(number) or {"width": 612.0, "height": 792.0, "words": []}
        fields = {}
        for section in pg["key_values"]["sections"]:
            for v in section["values"]:
                value = "" if is_empty(v.get("original_value")) else str(v["original_value"]).strip()
                raw = (v.get("value_bbox") or [[0, 0, 0, 0]])[0]
                bbox = normalize_bbox(raw, facts["width"], facts["height"]) if value else None
                # Gemini passes have word_index [] by design, so only ground truth is held to grounding.
                grounded = _grounded(v.get("word_index")) or v.get("model_selected") != ["M41"]
                valid = bool(v.get("valid", True))
                usable = bool(value) and grounded and valid and bbox is not None
                fields[v["key"]] = {
                    "value": value,
                    "bbox": bbox if usable else None,
                    "label": anchor_label(facts["words"], raw) if usable and facts["words"] else "",
                    "valid": valid,
                    "usable": usable,
                }
        tables = []
        for section in pg["tables"]["sections"]:
            for table in section["values"]:
                columns = [h["original_value"] for h in table["headers"]]
                rows = []
                for row in table["rows"]:
                    cells = {}
                    for column, cell in zip(columns, row["cells"]):
                        value = "" if is_empty(cell.get("original_value")) else str(cell["original_value"]).strip()
                        raw = (cell.get("value_bbox") or [[0, 0, 0, 0]])[0]
                        cells[column] = {"value": value,
                                         "bbox": normalize_bbox(raw, facts["width"], facts["height"]) if value else None}
                    rows.append(cells)
                tables.append({"type": table["original_value"], "rows": rows})
        pages.append({"number": number, "fields": fields, "tables": tables})
    vendor_name = next((p["fields"]["Vendor Name"]["value"] for p in pages
                        if p["fields"].get("Vendor Name", {}).get("value")), "")
    return {
        "doc_id": doc_id,
        "vendor_name": vendor_name,
        "vendor_key": vendor_key(vendor_name),
        "page_count": len(pages),
        "is_scanned": not any(f.get("has_text_layer") for f in page_facts.values()),
        "pages": pages,
    }


def fill_vector(page: dict) -> list[float]:
    """Structure signature of one page: 1 for every filled key-value entity and every table type present.
    A leading constant 1 keeps the vector non-zero, so cosine similarity is always defined."""
    vec = [1.0]
    vec += [0.0 if key in DEFAULTED_KEYS else float(bool(page["fields"].get(key, {}).get("value"))) for key in KV_KEYS]
    present = {t["type"] for t in page["tables"] if t["rows"]}
    vec += [float(table in present) for table in TABLE_TYPES]
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = sum(x * x for x in a) ** 0.5, sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Position bins and box ranges. Both tolerate the noise of an extracted box: a bin spreads a centre over the
# nearest grid cells, a range is a rectangle around the typical centre. Ranges are [x0, y0, x1, y1] page fractions.
# ---------------------------------------------------------------------------


def soft_bins(cx: float, cy: float, grid: int) -> list[float]:
    """Bilinear weights of one point over the centres of a grid x grid page grid (row-major, sums to 1). A centre
    near a cell edge is shared with the neighbouring cell, so a small shift changes the vector only a little."""
    def axis(v: float) -> dict[int, float]:
        if grid == 1:
            return {0: 1.0}
        t = min(max(v * grid - 0.5, 0.0), grid - 1.0)   # position in units of cell centres
        i = min(int(t), grid - 2)
        return {i: 1.0 - (t - i), i + 1: t - i}
    out = [0.0] * (grid * grid)
    for row, wy in axis(cy).items():
        for col, wx in axis(cx).items():
            out[row * grid + col] += wy * wx
    return out


def position_vector(page: dict, grid: int) -> list[float]:
    """Per entity (KV_KEYS order), the soft bins of its value box centre; all zero when it has no usable box."""
    vec = []
    for key in KV_KEYS:
        bbox = None if key in DEFAULTED_KEYS else page["fields"].get(key, {}).get("bbox")
        vec += soft_bins(*bbox_center(bbox), grid) if bbox else [0.0] * (grid * grid)
    return vec


def structure_vector(page: dict, grid: int, weight: float) -> list[float]:
    """fill_vector (which fields and tables) followed by the weighted position bins (roughly where)."""
    return fill_vector(page) + [weight * v for v in position_vector(page, grid)]


def box_range(cx: float, cy: float, margin: float) -> list[float]:
    return [max(cx - margin, 0.0), max(cy - margin, 0.0), min(cx + margin, 1.0), min(cy + margin, 1.0)]


def in_range(point: tuple[float, float] | None, rng: list[float] | None) -> bool:
    return bool(point and rng) and rng[0] <= point[0] <= rng[2] and rng[1] <= point[1] <= rng[3]


def range_area(rng: list[float]) -> float:
    return max(rng[2] - rng[0], 0.0) * max(rng[3] - rng[1], 0.0)


def range_overlap(a: list[float], b: list[float]) -> float:
    """Area of the intersection of two ranges."""
    return max(min(a[2], b[2]) - max(a[0], b[0]), 0.0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)


def range_to_bbox(rng: list[float]) -> list[float]:
    return [rng[0], rng[1], rng[2] - rng[0], rng[3] - rng[1]]


def normalize_log(log_scores: dict[str, float]) -> dict[str, float]:
    """Unnormalized log scores -> probabilities that sum to 1 (log-sum-exp, so tiny likelihoods don't underflow)."""
    top = max(log_scores.values())
    weights = {k: math.exp(v - top) for k, v in log_scores.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


# ---------------------------------------------------------------------------
# Compact prediction format (what Gemini returns) <-> output-schema envelope (what is saved to disk)
#   {"pages": [{"number": 1,
#               "fields": {"Invoice Number": {"value": "27236", "box_2d": [ymin, xmin, ymax, xmax]}, ...},
#               "tables": {"Item Line Table": [{"Item": "...", ...}], "Expense Table": [...]}}]}
# ---------------------------------------------------------------------------


def compact_to_envelope(prediction: dict, page_facts: dict[int, dict], model: str) -> list[dict]:
    """A Gemini prediction -> the full output schema, with boxes in page units, so a saved pass file has the same
    shape as ground truth. Scores are null (Gemini gives none) and word_index is [] (no token stream)."""
    by_number = {int(p.get("number", i + 1)): p for i, p in enumerate(prediction.get("pages", []))}
    envelope = []
    for number in sorted(page_facts):
        page = by_number.get(number, {})
        facts = page_facts[number]
        fields = page.get("fields") or {}
        sections = []
        for section, keys in SECTIONS.items():
            values = []
            for key in keys:
                field = fields.get(key) or {}
                value = "USD" if key == "Terms Currency" else (str(field.get("value") or "").strip() or " ")
                bbox = box2d_to_bbox(field.get("box_2d")) if value.strip() else None
                values.append({
                    "key": key, "display_key": key, "original_value": value,
                    "key_score": None, "value_score": None, "word_index": [],
                    "key_bbox": [[0, 0, 0, 0]],
                    "value_bbox": [denormalize_bbox(bbox, facts["width"], facts["height"])],
                    "model_selected": [model], "valid": True,
                })
            sections.append({"name": section, "order": 1, "values": values})
        tables = []
        for table_type, columns in TABLE_TYPES.items():
            rows = (page.get("tables") or {}).get(table_type) or []
            if not rows:
                continue
            tables.append({
                "original_value": table_type, "value_score": None, "value_bbox": [[0, 0, 0, 0]],
                "column_order": [], "row_order": [],
                "headers": [{"original_value": c, "value_score": None, "value_bbox": [[0, 0, 0, 0]],
                             "column_bbox": [[0, 0, 0, 0]]} for c in columns],
                "rows": [{
                    "cells": [{"original_value": str(row.get(c) or "").strip() or " ", "value_score": None,
                               "value_bbox": [[0, 0, 0, 0]], "valid": True} for c in columns],
                    "original_value": "", "value_score": None, "value_bbox": [[0, 0, 0, 0]],
                    "row_bbox": [[0, 0, 0, 0]], "word_index": [[] for _ in columns],
                } for row in rows],
            })
        envelope.append({
            "classtype": DOC_TYPE, "identifier": "", "number": number,
            "key_values": {"sections": sections},
            "tables": {"sections": [{"name": "Tables", "order": 1, "values": tables}] if tables else []},
        })
    return envelope


# ---------------------------------------------------------------------------
# Gemini: response schema, prompts, call plumbing
# ---------------------------------------------------------------------------

_BOX_SCHEMA = types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER),
                           description="[ymin, xmin, ymax, xmax] on a 0-1000 page grid; [] when the value is empty")


def extraction_schema() -> types.Schema:
    """Constrained-decoding schema for one document."""
    field = types.Schema(type=types.Type.OBJECT, properties={
        "value": types.Schema(type=types.Type.STRING), "box_2d": _BOX_SCHEMA}, required=["value", "box_2d"])
    kv_keys = [k for k in KV_KEYS if k not in DEFAULTED_KEYS]
    tables = types.Schema(type=types.Type.OBJECT, properties={
        table: types.Schema(type=types.Type.ARRAY, items=types.Schema(
            type=types.Type.OBJECT,
            properties={c: types.Schema(type=types.Type.STRING) for c in columns}, required=columns))
        for table, columns in TABLE_TYPES.items()})
    page = types.Schema(type=types.Type.OBJECT, properties={
        "number": types.Schema(type=types.Type.INTEGER),
        "fields": types.Schema(type=types.Type.OBJECT, properties={k: field for k in kv_keys}, required=kv_keys),
        "tables": tables,
    }, required=["number", "fields", "tables"])
    return types.Schema(type=types.Type.OBJECT, properties={"pages": types.Schema(type=types.Type.ARRAY, items=page)},
                        required=["pages"])


def catalog_text() -> str:
    lines = ["NORMALIZATION RULES"] + [f"  {name:<9} {rule}" for name, rule in NORMALIZATION_RULES.items()]
    lines += ["", "KEY-VALUE FIELDS (one set per page)"]
    for section, keys in SECTIONS.items():
        for key in keys:
            if key not in DEFAULTED_KEYS:
                lines.append(f"  [{section}] {key} ({VALUE_TYPES[key]}): {DESCRIPTIONS[key]}")
    for table, columns in TABLE_TYPES.items():
        lines.append(f"TABLE {table}: columns " + ", ".join(f"{c} ({VALUE_TYPES[c]})" for c in columns))
    lines += ["", f"INVARIANT: {ARITHMETIC_INVARIANT}"]
    return "\n".join(lines)


EXTRACT_PROMPT = """You are an OCR and information-extraction engine for invoices.
Return one entry in "pages" per PDF page, in page order. On each page, give every key-value field: its value as
printed (normalized per the rules) and box_2d, the box around the printed value as [ymin, xmin, ymax, xmax] on a
0-1000 grid of the page. A field not printed on a page gets value "" and box_2d []. Page 2 onward fills header
fields only when that page re-prints them. "Item Line Table" holds product/service lines and "Expense Table" holds
invoice-level charges (tax, shipping, fees, discounts); use [] for a table not on the page.

{catalog}
"""

GUIDED_SUFFIX = """
MEMORY GUIDE
The rules below come from a memory of earlier invoices: where each field's value is usually printed and which
label usually sits next to it. Use them to find fields you would otherwise miss and to place boxes precisely.
Never copy a value from the memory; always read it from this PDF. If a rule disagrees with what is printed,
trust the PDF.

{rules}
"""

GATED_SUFFIX = """
MEMORY GUIDE WITH CONFIDENCES
The rules below come from a memory of earlier invoices. Each carries a probability, computed from how well this
PDF matches the remembered layout and how consistently the value sat in the same region on earlier invoices:
- "search region box_2d [...] (confidence p)": the value is printed inside that region with probability p. Look
  there first, but return the box of the value as printed, not the region.
- "position uncertain": the remembered position is not trustworthy for this PDF; only the label hint is given.
- "expected with probability p": how likely it is that the field is printed on this page at all.
Rules that are too unlikely to apply are left out. Never copy a value from the memory; always read it from this
PDF. If a rule disagrees with what is printed, trust the PDF.

{rules}
"""



def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, errors.APIError) and exc.code in (429, 500, 502, 503, 504)


with_retries = retry(retry=retry_if_exception(_is_transient), wait=wait_random_exponential(multiplier=1, max=60),
                     stop=stop_after_attempt(8), reraise=True)


def parse_json_response(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        context = cleaned[max(exc.pos - 150, 0):exc.pos + 50]
        raise ValueError(f"Model did not return valid JSON ({exc.msg} at char {exc.pos}), near: {context!r}") from exc


def gemini_extract(client, model: str, pdf_path: Path, system: str, schema: types.Schema) -> dict:
    """One constrained-JSON Gemini call on a PDF at temperature 0, retried on transient Vertex errors."""
    config = types.GenerateContentConfig(system_instruction=system, temperature=0,
                                         response_mime_type="application/json", response_schema=schema)
    part = types.Part.from_bytes(data=Path(pdf_path).read_bytes(), mime_type="application/pdf")
    response = with_retries(client.models.generate_content)(
        model=model, contents=[types.Content(role="user", parts=[part])], config=config)
    return parse_json_response(response.text or "")


# ---------------------------------------------------------------------------
# Table row alignment (scoring)
# ---------------------------------------------------------------------------


def align_rows(gt_rows: list[dict], pred_rows: list[dict]) -> list[tuple[dict, dict | None]]:
    """Greedy one-to-one matching of ground-truth rows to predicted rows by mean cell similarity.
    Rows are {column: value}. Unmatched ground-truth rows pair with None."""
    pairs, used = [], set()
    scored = sorted(
        ((sum(string_similarity(g.get(c), p.get(c)) for c in g) / max(len(g), 1), gi, pi)
         for gi, g in enumerate(gt_rows) for pi, p in enumerate(pred_rows)), reverse=True)
    matched = {}
    for score, gi, pi in scored:
        if gi not in matched and pi not in used and score > 0.3:
            matched[gi] = pi
            used.add(pi)
    for gi, g in enumerate(gt_rows):
        pairs.append((g, pred_rows[matched[gi]] if gi in matched else None))
    return pairs
