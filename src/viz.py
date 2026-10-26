"""Pictures for the demo: Neo4j subgraphs rendered inline with neo4j-viz, and matplotlib views of pages and of the
gate's numbers. Colors follow a fixed role order (never cycled), text stays in neutral ink."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
from matplotlib.patches import Rectangle
from neo4j_viz import Node, Relationship, VisualizationGraph
from neo4j_viz.neo4j import from_neo4j

import helpers as H

# --- Palette (validated categorical order; chart chrome) -------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

LABEL_COLORS = {"DocumentType": INK_2, "Vendor": BLUE, "PageLayout": ORANGE, "SectionLayout": AQUA,
                "EntityLayout": YELLOW, "TableLayout": MAGENTA, "ColumnLayout": GREEN,
                "NormalizationRule": VIOLET, "Invariant": RED}
GATE_COLORS = {"box": BLUE, "label": YELLOW, "withheld": MUTED, "no_position": GRID}
STRENGTH_COLORS = {"strong": BLUE, "weak": AQUA, "single": YELLOW, "none": GRID}
CAPTION_PROPERTY = {"DocumentType": "name", "Vendor": "key", "PageLayout": "id", "SectionLayout": "name",
                    "EntityLayout": "key", "TableLayout": "type", "ColumnLayout": "header",
                    "NormalizationRule": "name", "Invariant": "name"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8, "font.size": 9,
})


# --- Graphs --------------------------------------------------------------------------------------------------


def _label(node: Node) -> str:
    return node.properties["labels"][0]


def default_caption(node: Node) -> str:
    """The node's own name: a vendor's key, a layout's page and variant, an entity's key..."""
    label = _label(node)
    value = node.properties.get(CAPTION_PROPERTY.get(label, ""), label)
    if label == "PageLayout":
        return f"page {node.properties['page_number']} · v{node.properties['variant']} ({node.properties['doc_count']} docs)"
    return str(value)


def show_graph(driver, query: str, *, caption: Callable[[Node], str] = default_caption,
               color: Callable[[Node], str] | None = None, size: Callable[[Node], float] | None = None,
               layout: str = "forcedirected", height: str = "520px", **params):
    """Run a Cypher query that returns nodes/relationships/paths and render the subgraph inline.
    `color`/`size`/`caption` map each node to its style; nodes are colored by label by default."""
    vg = from_neo4j(driver.execute_query(query, **params))
    for n in vg.nodes:
        n.caption = caption(n)
        n.color = color(n) if color else LABEL_COLORS.get(_label(n), MUTED)
        n.size = size(n) if size else 14 if _label(n) in ("Vendor", "PageLayout") else 9
    for r in vg.relationships:
        r.caption = r.properties.get("type")
    return vg.render(layout=layout, height=height)


def show_virtual(nodes: list[dict], edges: list[dict], layout: str = "forcedirected", height: str = "460px"):
    """Render a graph that is not in Neo4j (e.g. a test page and its candidate layouts).
    nodes: {id, caption, color, size}; edges: {source, target, caption, width?}."""
    vg = VisualizationGraph(
        [Node(id=n["id"], caption=n["caption"], color=n.get("color", MUTED), size=n.get("size", 12)) for n in nodes],
        [Relationship(source=e["source"], target=e["target"], caption=e.get("caption"),
                      width=e.get("width", 1)) for e in edges])
    return vg.render(layout=layout, height=height)


def legend_html(colors: dict[str, str]) -> str:
    """A small HTML legend to print above a neo4j-viz graph (identity is never color alone)."""
    items = "".join(f'<span style="margin-right:14px;white-space:nowrap"><span style="display:inline-block;'
                    f'width:10px;height:10px;border-radius:50%;background:{c};margin-right:5px"></span>{k}</span>'
                    for k, c in colors.items())
    return f'<div style="font:12px system-ui,sans-serif;color:{INK_2};margin:4px 0 6px">{items}</div>'


# --- Pages ---------------------------------------------------------------------------------------------------


def render_page(pdf_path: Path, page_number: int, zoom: float = 1.4) -> np.ndarray:
    with pymupdf.open(pdf_path) as pdf:
        pix = pdf[page_number - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]


def page_axes(pdf_path: Path, page_number: int, ax=None, title: str | None = None, fade: float = 0.0):
    """Draw a PDF page in page-fraction coordinates (0..1, top-left origin) and return the axes."""
    img = render_page(pdf_path, page_number)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 6.2 * img.shape[0] / img.shape[1]))
    ax.imshow(img, extent=(0, 1, 1, 0))
    if fade:
        ax.add_patch(Rectangle((0, 0), 1, 1, color="white", alpha=fade, zorder=1))
    ax.set_xlim(0, 1), ax.set_ylim(1, 0)
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True), s.set_color(AXIS)
    if title:
        ax.set_title(title, loc="left", pad=16)
    return ax


def draw_box(ax, bbox, color: str, label: str | None = None, lw: float = 1.6, ls: str = "-", fill: bool = False,
             alpha: float = 1.0):
    x, y, w, h = bbox
    ax.add_patch(Rectangle((x, y), w, h, fill=fill, edgecolor=color, facecolor=color if fill else "none",
                           lw=lw, ls=ls, alpha=alpha, zorder=3))
    if label:
        ax.text(x, y - 0.004, label, color=INK, fontsize=6.5, va="bottom", zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.8})


def draw_range(ax, rng, color: str, label: str | None = None, ls: str = "--"):
    draw_box(ax, H.range_to_bbox(rng), color, label, lw=1.4, ls=ls)


def page_with_fields(pdf_path: Path, page: dict, ax=None, title: str | None = None, keys=None, color: str = BLUE):
    """A page with its ground-truth value boxes, labelled by entity."""
    ax = page_axes(pdf_path, page["number"], ax, title)
    for key, f in page["fields"].items():
        if f["bbox"] and (keys is None or key in keys):
            draw_box(ax, f["bbox"], color, key)
    return ax


# --- Charts --------------------------------------------------------------------------------------------------


def term_bars(explain: pd.DataFrame, ax=None, title: str = "What moves the belief: log-terms per candidate"):
    """Grouped horizontal bars: each Bayes term's log contribution for the top candidates (relative to the best
    candidate, so 0 = as good as the winner on that term)."""
    terms = ["prior", "name", "fields", "boxes"]
    rel = explain[terms] - explain[terms].max()
    labels = [f"{i if i == 'NEW' else explain.at[i, 'vendor']} · P={explain.at[i, 'posterior']:.2f}"
              for i in explain.index]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.55 * len(explain) + 1.2))
    y = np.arange(len(explain))
    h = 0.8 / len(terms)
    for j, (term, color) in enumerate(zip(terms, (MUTED, BLUE, ORANGE, AQUA))):
        ax.barh(y + (j - 1.5) * h, rel[term].clip(lower=-40), height=h * 0.92, color=color, label=term)
    ax.set_yticks(y, labels), ax.invert_yaxis()
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("log-score minus the best candidate's (clipped at −40)")
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title(title, loc="left", pad=22)
    return ax


def reliability_plot(groups: dict[str, pd.DataFrame], ax=None, title: str = "Is the confidence honest?",
                     min_n: int = 5):
    """Predicted confidence against observed hit rate, one line per group; marker area ∝ number of rules. Bins
    with fewer than min_n rules are left out (their hit rate is noise)."""
    from src.evaluation import reliability
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--", label="calibrated")
    for (name, rules), color in zip(groups.items(), (BLUE, ORANGE, AQUA)):
        if rules.empty:
            continue
        t = reliability(rules.confidence, rules.hit)
        t = t[t.n >= min_n]
        ax.plot(t.mean_predicted, t.observed, color=color, lw=2, label=name, zorder=3)
        ax.scatter(t.mean_predicted, t.observed, s=12 + 1.5 * np.sqrt(t.n) * 8, color=color,
                   edgecolor=SURFACE, lw=1.5, zorder=4)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="predicted confidence", ylabel="observed hit rate")
    ax.legend(loc="upper left")
    ax.set_title(title, loc="left")
    return ax


def coverage_plot(groups: dict[str, pd.DataFrame], ax=None, gate_box: float | None = None,
                  title: str = "Risk–coverage: raising the threshold"):
    """Accuracy of the rules shown against the share shown, as the threshold τ rises."""
    from src.evaluation import coverage_table
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.4))
    lowest = 1.0
    for (name, rules), color in zip(groups.items(), (BLUE, ORANGE, AQUA)):
        if rules.empty:
            continue
        t = coverage_table(rules).dropna(subset=["accuracy_of_shown"])
        lowest = min(lowest, t.accuracy_of_shown.min())
        ax.plot(t.coverage, t.accuracy_of_shown, color=color, lw=2, marker="o", ms=4, label=name)
        if gate_box is not None and (t.tau == round(gate_box, 1)).any():
            row = t[t.tau == round(gate_box, 1)].iloc[0]
            ax.annotate(f"τ={gate_box}", (row.coverage, row.accuracy_of_shown), textcoords="offset points",
                        xytext=(6, -12), fontsize=7.5, color=INK_2)
    ax.set(xlim=(0, 1.02), ylim=(max(0.0, lowest - 0.1), 1.01),
           xlabel="coverage (share of rules shown with a position)", ylabel="accuracy of shown rules")
    ax.legend(loc="lower left")
    ax.set_title(title, loc="left")
    return ax


def gate_mix_plot(summary: pd.DataFrame, ax=None, title: str = "What the gate did, per scenario"):
    """Stacked horizontal bars of box / label / withheld shares, one bar per scenario, with the hit rate of
    shown boxes written beside each bar."""
    df = summary.reset_index()
    labels = [f"{s}" for s in df.scenario]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.42 * len(df) + 1.2))
    left = np.zeros(len(df))
    for col, gate in (("share_box", "box"), ("share_label", "label"), ("share_withheld", "withheld")):
        ax.barh(labels, df[col], left=left, color=GATE_COLORS[gate], edgecolor=SURFACE, lw=2, height=0.7,
                label=gate)
        left += df[col].to_numpy()
    for i, (hb, mapacc) in enumerate(zip(df.hit_box, df.map_acc)):
        txt = f"MAP right {mapacc:.0%}" + ("" if pd.isna(hb) else f" · boxes right {hb:.0%}")
        ax.text(1.01, i, txt, va="center", fontsize=7.5, color=INK_2)
    ax.set_xlim(0, 1), ax.invert_yaxis(), ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of rules considered")
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title(title, loc="left", pad=22)
    return ax
