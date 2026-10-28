"""Every setting of the demo, with the same values as gating.ipynb's config cell. A frozen dataclass, so a sweep
can vary one setting with `dataclasses.replace(cfg, field_temperature=2.0)` without touching the others."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # --- Neo4j ---------------------------------------------------------------------------------------------------
    neo4j_uri: str = "bolt://localhost:7688"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "gating-local"

    # --- Gemini (src/llm.py) -------------------------------------------------------------------------------------
    gcp_project: str = "rahhh"
    gcp_location: str = "global"
    ocr_model: str = "gemini-2.5-flash"
    gated_pass_name: str = "rule_groups_gated"   # results/<name>/ for the gated pass with rule-group guidance

    # --- Data ----------------------------------------------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path.cwd().parent / "data")   # holds input/ (PDFs), output/ (GT)
    results_dir: Path = Path("results")         # saved Gemini passes from gating.ipynb (only Part IV reads them)

    # --- Split ---------------------------------------------------------------------------------------------------
    train_fraction: float = 2 / 3               # share of each vendor's documents used to build the graph
    split_seed: int = 7

    # --- Consolidation -------------------------------------------------------------------------------------------
    structure_similarity: float = 0.85          # fill-vector cosine at or above which two pages share a layout variant
    rule_group_similarity: float = 0.85         # structure-vector cosine at or above which a page is a rule group's candidate
    bbox_tolerance: float = 0.03                # box centres within this many page-widths/heights are "the same spot"
    min_support_fraction: float = 0.6           # share of located observations that must agree to make a rule
    min_support_docs: int = 2                   # ...and at least this many documents
    strong_support_docs: int = 3                # a rule backed by this many documents is "strong"

    # --- Position bins and box ranges ----------------------------------------------------------------------------
    position_grid: int = 4                      # 4 x 4 grid for the position part of the structure vector
    position_weight: float = 1.0                # weight of the position part against the fill part
    box_range_min: float = 0.06                 # smallest half-width of a rule's range
    box_range_max: float = 0.15                 # largest half-width
    box_range_spread_k: float = 2.0             # otherwise the half-width is this many bbox_spreads

    # --- Posterior over layouts ----------------------------------------------------------------------------------
    field_smoothing_m: float = 2.0              # pseudo-pages of corpus behaviour mixed into every layout's fill rates
    field_miss: float = 0.05                    # P(extraction leaves a printed field empty)
    field_false_fill: float = 0.02              # P(extraction fills a field that is not printed)
    name_error: float = 0.10                    # P(the extracted vendor name maps to the wrong vendor key)
    name_variant: float = 0.5                   # share of those errors that are a variant spelling of the true key
    field_temperature: float = 1.0              # > 1 flattens the field term (fields are not independent)
    rule_prior_strength: float = 2.0            # pseudo-trials of the empirical-Bayes prior on p_in_range

    # --- Gate ----------------------------------------------------------------------------------------------------
    gate_box: float = 0.7                       # confidence at or above which a rule's typical box is shown
    gate_label: float = 0.4                     # ...at or above which only its label is shown; below, withheld

    # --- Generated test pages (the stand-in for Gemini's unguided call) ------------------------------------------
    scenario_seed: int = 11
    noise_box_miss: float = 0.057               # mean distance a simulated box centre lands from the true one
    noise_field_miss: float = 0.05              # P(a printed field is dropped)
    noise_false_fill: float = 0.02              # P(an unprinted field is filled, at a random spot)
    noise_name_error: float = 0.10              # P(the vendor name is misread: variant spelling or typo)
    twin_min_similarity: float = 0.9            # structure cosine at or above which two vendors' layouts are "twins"
    twin_min_shared_rules: int = 3              # ...and they must both locate at least this many entities
    drift_shift: tuple[float, float] = (0.0, 0.25)  # (dx, dy) applied to every box of a "redesigned" page

    @property
    def input_dir(self) -> Path:
        return self.data_dir / "input"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"
