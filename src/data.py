"""The ground-truth corpus and its vendor-proportional train/test split (gating.ipynb, "Load the ground truth")."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

import helpers as H
from src.config import Config

NEW = "NEW"   # the class "a vendor or layout the memory has never seen"
BUCKET_ORDER = ["0 (unseen)", "1", "2-3", "4-9", "10+"]


def size_bucket(n_train: int) -> str:
    return "0 (unseen)" if n_train == 0 else "1" if n_train == 1 else "2-3" if n_train <= 3 else \
           "4-9" if n_train <= 9 else "10+"


def proportional_split(records: dict, fraction: float, seed: int) -> tuple[dict[str, int], set[str], set[str]]:
    """Training count per vendor (largest-remainder rounding to an exact corpus total), then a seeded random draw
    of which documents within each vendor go to training. Returns (n_train per vendor, train ids, test ids)."""
    rng = random.Random(seed)
    by_vendor = defaultdict(list)
    for doc_id in sorted(records):
        by_vendor[records[doc_id]["vendor_key"]].append(doc_id)
    ideal = {v: fraction * len(ids) for v, ids in by_vendor.items()}
    n_train = {v: int(x) for v, x in ideal.items()}
    leftover = round(fraction * len(records)) - sum(n_train.values())
    tie_break = {v: rng.random() for v in by_vendor}
    for v in sorted(ideal, key=lambda v: (ideal[v] - n_train[v], tie_break[v]), reverse=True)[:leftover]:
        n_train[v] += 1
    train, test = set(), set()
    for v, ids in sorted(by_vendor.items()):
        chosen = set(rng.sample(ids, n_train[v]))
        train |= chosen
        test |= set(ids) - chosen
    return n_train, train, test


@dataclass
class Corpus:
    """Flat ground-truth records (`helpers.flatten_document`), the PDF facts they were read with, and the split."""
    cfg: Config
    records: dict[str, dict]
    page_facts: dict[str, dict]
    train_ids: set[str]
    test_ids: set[str]

    @classmethod
    def load(cls, cfg: Config) -> "Corpus":
        doc_ids = sorted(p.stem for p in cfg.output_dir.glob("*.json") if (cfg.input_dir / f"{p.stem}.pdf").exists())
        assert doc_ids, f"no ground-truth JSON with a matching PDF under {cfg.data_dir.resolve()}: set data_dir"
        facts = {d: H.read_pdf_page_facts(cfg.input_dir / f"{d}.pdf") for d in doc_ids}
        records = {d: H.document_to_record(d, json.loads((cfg.output_dir / f"{d}.json").read_text()), facts[d])
                   for d in doc_ids}
        _, train, test = proportional_split(records, cfg.train_fraction, cfg.split_seed)
        assert not train & test and len(train) + len(test) == len(records)
        return cls(cfg, records, facts, train, test)

    @property
    def train_count(self) -> Counter:
        return Counter(self.records[d]["vendor_key"] for d in self.train_ids)

    def vendor_of(self, doc_id: str) -> str:
        return self.records[doc_id]["vendor_key"]

    def truth(self, doc_id: str) -> str:
        """What the posterior should find: the document's vendor if the graph has seen it, otherwise NEW."""
        vendor = self.vendor_of(doc_id)
        return vendor if self.train_count[vendor] else NEW

    def bucket(self, vendor: str) -> str:
        return size_bucket(self.train_count[vendor])

    def pdf(self, doc_id: str):
        return self.cfg.input_dir / f"{doc_id}.pdf"

    def vendor_table(self) -> pd.DataFrame:
        """One row per vendor root: documents, train/test counts, training-set size bucket, name variants."""
        rows = defaultdict(lambda: {"names": set(), "n_docs": 0, "n_train": 0, "n_test": 0})
        for doc_id, rec in self.records.items():
            row = rows[rec["vendor_key"]]
            row["names"].add(rec["vendor_name"] or "(none)")
            row["n_docs"] += 1
            row["n_train" if doc_id in self.train_ids else "n_test"] += 1
        df = pd.DataFrame([{"vendor_key": k, "name_variants": " | ".join(sorted(v["names"])),
                            "n_docs": v["n_docs"], "n_train": v["n_train"], "n_test": v["n_test"],
                            "bucket": size_bucket(v["n_train"])} for k, v in rows.items()])
        return df.sort_values(["n_docs", "vendor_key"], ascending=[False, True]).reset_index(drop=True)

    def bucket_table(self) -> pd.DataFrame:
        return self.vendor_table().groupby("bucket").agg(
            vendors=("vendor_key", "count"), train_docs=("n_train", "sum"), test_docs=("n_test", "sum")
        ).reindex(BUCKET_ORDER).fillna(0).astype(int)
