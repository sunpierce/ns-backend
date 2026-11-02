#!/usr/bin/env python3
"""
Load the Neurosynth parquet files into PostgreSQL (schema `ns`).

    pip install -r requirements-load.txt
    python create_db.py --url "postgresql://USER:PASSWORD@HOST:5432/DBNAME" --data-dir ./data

Expects coordinates.parquet, metadata.parquet and annotations.parquet
(the files in https://github.com/spiketren/ns-nano) in --data-dir.

Tables
------
ns.metadata          (study_id PK, contrast_id, authors, journal, year, title, fts tsvector + GIN)
ns.coordinates       (study_id, x, y, z)  + B-tree indexes on x, y, z and study_id
ns.annotations_terms (term, study_id, weight)  sparse TF-IDF, PK (term, study_id) + index on study_id
ns.terms             (term PK, n_studies)  vocabulary, for validation and suggestions

Coordinates are kept as plain x/y/z columns (MNI millimetres), so no PostGIS is needed. A sphere query
filters with a box on the indexed columns first and then applies the exact Euclidean test.
"""
import argparse
import io
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text

S = "ns"


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.1f}s] {msg}", flush=True)


def copy_df(engine, table, df, columns):
    """Bulk-load a DataFrame with COPY (much faster than INSERT)."""
    buf = io.StringIO()
    df[columns].to_csv(buf, sep="\t", header=False, index=False, na_rep="\\N")
    buf.seek(0)
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.copy_expert(f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT text)", buf)
        raw.commit()
    finally:
        raw.close()


def build_metadata(engine, path):
    meta = pd.read_parquet(path)
    meta["study_id"] = meta["study_id"].astype(str)
    meta["contrast_id"] = meta["contrast_id"].astype(str)
    meta["year"] = pd.to_numeric(meta["year"], errors="coerce").astype("Int64")
    for c in ("authors", "journal", "title"):
        meta[c] = meta[c].astype(str).str.replace(r"[\t\n\r\\]", " ", regex=True)
    meta = meta.drop_duplicates("study_id")
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {S}.metadata CASCADE"))
        conn.execute(text(f"""
            CREATE TABLE {S}.metadata (
                study_id    TEXT PRIMARY KEY,
                contrast_id TEXT,
                authors     TEXT,
                journal     TEXT,
                year        INTEGER,
                title       TEXT,
                fts tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(title, '') || ' ' || coalesce(authors, '') || ' ' || coalesce(journal, ''))
                ) STORED
            )"""))
    copy_df(engine, f"{S}.metadata", meta, ["study_id", "contrast_id", "authors", "journal", "year", "title"])
    with engine.begin() as conn:
        conn.execute(text(f"CREATE INDEX idx_metadata_fts ON {S}.metadata USING GIN (fts)"))
        conn.execute(text(f"ANALYZE {S}.metadata"))
    log(f"metadata: {len(meta):,} studies")


def build_coordinates(engine, path):
    c = pd.read_parquet(path, columns=["study_id", "x", "y", "z"])
    c["study_id"] = c["study_id"].astype(str)
    c = c[np.isfinite(c[["x", "y", "z"]]).all(axis=1)]
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {S}.coordinates CASCADE"))
        conn.execute(text(f"""
            CREATE TABLE {S}.coordinates (
                id       BIGSERIAL PRIMARY KEY,
                study_id TEXT NOT NULL,
                x DOUBLE PRECISION NOT NULL,
                y DOUBLE PRECISION NOT NULL,
                z DOUBLE PRECISION NOT NULL
            )"""))
    copy_df(engine, f"{S}.coordinates", c, ["study_id", "x", "y", "z"])
    with engine.begin() as conn:
        for col in ("study_id", "x", "y", "z"):
            conn.execute(text(f"CREATE INDEX idx_coordinates_{col} ON {S}.coordinates ({col})"))
        conn.execute(text(f"ANALYZE {S}.coordinates"))
    log(f"coordinates: {len(c):,} peaks")


def build_annotations(engine, path, batch=200):
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    term_cols = [n for n in names if n.startswith("terms_")]
    sid = pd.read_parquet(path, columns=["study_id"])["study_id"].astype(str).to_numpy()
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {S}.annotations_terms CASCADE"))
        conn.execute(text(f"DROP TABLE IF EXISTS {S}.terms CASCADE"))
        conn.execute(text(f"""
            CREATE UNLOGGED TABLE {S}.annotations_terms (
                term     TEXT NOT NULL,
                study_id TEXT NOT NULL,
                weight   REAL NOT NULL
            )"""))
    total = 0
    for i in range(0, len(term_cols), batch):          # read a few hundred columns at a time to limit RAM
        cols = term_cols[i:i + batch]
        block = pd.read_parquet(path, columns=cols).to_numpy(dtype=np.float32)
        r, k = np.nonzero(np.nan_to_num(block) > 0)     # keep only non-zero weights (sparse)
        terms = np.array([c.split("__", 1)[-1].strip().lower() for c in cols])
        df = pd.DataFrame({"term": terms[k], "study_id": sid[r], "weight": block[r, k]})
        df = df.groupby(["term", "study_id"], as_index=False)["weight"].max()   # guard against duplicate names
        copy_df(engine, f"{S}.annotations_terms", df, ["term", "study_id", "weight"])
        total += len(df)
        log(f"annotations: columns {i + 1}-{i + len(cols)} of {len(term_cols)} ({total:,} rows)")
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {S}.annotations_terms SET LOGGED"))
        conn.execute(text(f"ALTER TABLE {S}.annotations_terms ADD PRIMARY KEY (term, study_id)"))
        conn.execute(text(f"CREATE INDEX idx_annotations_study ON {S}.annotations_terms (study_id)"))
        conn.execute(text(f"""
            CREATE TABLE {S}.terms AS
            SELECT term, COUNT(*) FILTER (WHERE weight >= 0.001)::int AS n_studies
            FROM {S}.annotations_terms GROUP BY term"""))
        conn.execute(text(f"ALTER TABLE {S}.terms ADD PRIMARY KEY (term)"))
        conn.execute(text(f"ANALYZE {S}.annotations_terms"))
        conn.execute(text(f"ANALYZE {S}.terms"))
    log(f"annotations: {total:,} non-zero (term, study) weights, {len(term_cols):,} terms")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.getenv("DB_URL"), help="postgresql://USER:PASSWORD@HOST:5432/DBNAME")
    ap.add_argument("--data-dir", default=".", help="folder with the three parquet files")
    args = ap.parse_args()
    if not args.url:
        ap.error("pass --url or set DB_URL")
    url = args.url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(url, pool_pre_ping=True)
    with engine.begin() as conn:
        log("connected: " + conn.execute(text("SELECT version()")).scalar().split(",")[0])
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {S}"))
    d = args.data_dir
    build_metadata(engine, os.path.join(d, "metadata.parquet"))
    build_coordinates(engine, os.path.join(d, "coordinates.parquet"))
    build_annotations(engine, os.path.join(d, "annotations.parquet"))
    log("done ✔")


if __name__ == "__main__":
    main()
