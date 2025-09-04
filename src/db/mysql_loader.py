# src/db/mysql_loader.py
"""
Load processed parquet products into MySQL.

What it does (mirrors your 02 notebook):
- Build engine from config.py MYSQL
- Read headphones + laptops parquet from PROC_DIR
- Align to 12-column MySQL schema
- Normalize ASIN (upper), drop duplicates
- Either TRUNCATE + INSERT (simple) or UPSERT (optional)
- Return simple counts for sanity

Usage in 02 notebook:
    from src.db.mysql_loader import get_engine, truncate_and_load, upsert_load
    eng = get_engine()
    truncate_and_load(eng)   # or: upsert_load(eng)
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import MYSQL, PROC_DIR

# the exact column order based on MySQL table
EXPECTED_COLS = [
    "asin","title","brand","category_path","filtered_category",
    "price","average_rating","rating_number","description",
    "image_url","price_bucket","search_text"
]

#Build a SQLAlchemy engine from values in config.MYSQL
def get_engine() -> Engine:
    uri = f"mysql+pymysql://{MYSQL['user']}:{MYSQL['password']}@{MYSQL['host']}:{MYSQL['port']}/{MYSQL['db']}"
    return create_engine(uri, pool_pre_ping=True, future=True)

#Read a parquet file and align/clean it to EXPECTED_COLS."""
def _read_one(fp: Path) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(f"Missing parquet: {fp}")
    df = pd.read_parquet(fp)

    # keep only expected columns (drops extras like 'popularity' if present)
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{fp.name} missing columns: {missing}")
    df = df[EXPECTED_COLS].copy()

    # normalize
    df["asin"] = df["asin"].astype(str).str.strip().str.upper()
    for col in ["title","brand","category_path","filtered_category","price_bucket",
                "description","image_url","search_text"]:
        df[col] = df[col].astype(str).fillna("").str.strip()

    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0).astype(float)
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce").fillna(0).astype(float)
    df["rating_number"] = pd.to_numeric(df["rating_number"], errors="coerce").fillna(0).astype(int)

    return df

#Read both category parquets and drop duplicate ASINs
def _read_both_parquets() -> pd.DataFrame:
    p_head = PROC_DIR / "products_headphones_2023.parquet"
    p_lap  = PROC_DIR / "products_laptops_2023.parquet"
    df = pd.concat([_read_one(p_head), _read_one(p_lap)], ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["asin"], keep="first").reset_index(drop=True)
    print(f"[combine] rows: {before:,} → after ASIN dedupe: {len(df):,}")
    return df

#TRUNCATE TABLE products; then bulk INSERT all rows from the parquets.Returns (inserted_rows, table_count)
def truncate_and_load(engine: Engine, chunksize: int = 2000) -> Tuple[int, int]:
    df = _read_both_parquets()

    # truncate
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE products;"))
    print("[load] table truncated")

    # insert
    df.to_sql(
        "products",
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=chunksize
    )
    print(f"[load] inserted {len(df):,} rows")

    # verify
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM products;")).scalar()
    print("[verify] table row count:", total)
    return len(df), int(total)

#Returns (upserted_rows, table_count_after)
def upsert_load(engine: Engine, batch: int = 1000) -> Tuple[int, int]:
    df = _read_both_parquets()

    cols = EXPECTED_COLS
    col_list = ", ".join(cols)
    placeholders = ", ".join([f":{c}" for c in cols])
    update_clause = ", ".join([f"{c}=VALUES({c})" for c in cols if c != "asin"])  # don't update PK

    sql = text(f"""
        INSERT INTO products ({col_list})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_clause}
    """)

    records = df[cols].to_dict(orient="records")
    count = 0
    with engine.begin() as conn:
        for i in range(0, len(records), batch):
            conn.execute(sql, records[i:i+batch])
            count += len(records[i:i+batch])
    print(f"[upsert] upserted {count:,} rows")

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM products;")).scalar()
    print("[verify] table row count:", total)
    return count, int(total)
