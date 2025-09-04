#Build and maintain the Chroma vector index for products.
# - Persistent Chroma store under vector_store/
# - Pages products from MySQL in stable order (by ASIN)
# - Skips ASINs that are already indexed
# - Embeds search_text via OpenAI (batched)
# - Adds vectors + light metadata to Chroma
# - Simple semantic_search for quick testing

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Iterable
import math, time

import pandas as pd
from sqlalchemy import text
from openai import OpenAI
import chromadb

from config import PROJECT_ROOT, OPENAI_API_KEY
from src.db.mysql_loader import get_engine  # re-use database connection builder

# Configuration
COLLECTION_NAME = "products_2023_electronics"
VECTOR_DIR      = PROJECT_ROOT / "vector_store"
EMBED_MODEL     = "text-embedding-3-small"

## Batch size (tune as needed)
DEFAULT_SQL_PAGE   = 1000   # rows pulled from MySQL per page
DEFAULT_EMBED_BATCH= 200    # texts per OpenAI embeddings call
DEFAULT_GET_PAGE   = 5000   # ids fetched per Chroma .get() page


# Chroma + OpenAI clients 
def get_collection():
    VECTOR_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)

def _openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


# Helpers: fetching + preprocessing data
# Counts total number of products in the MySQL table
def _count_total_products() -> int:
    eng = get_engine()
    with eng.connect() as conn:
        return int(conn.execute(text("SELECT COUNT(*) FROM products")).scalar())

#Return a stable slice of products needed for embeddings + metadata
def _fetch_products_page(limit: int, offset: int) -> pd.DataFrame:
    eng = get_engine()
    q = text("""
        SELECT asin, title, brand, price, average_rating, rating_number,
               filtered_category, price_bucket, search_text
        FROM products
        ORDER BY asin
        LIMIT :lim OFFSET :off
    """)
    with eng.connect() as conn:
        df = pd.read_sql(q, conn, params={"lim": limit, "off": offset})
        
    # normalize ASINs & text
    df["asin"] = df["asin"].astype(str).str.strip().str.upper()
    df["search_text"] = (
        df["search_text"].fillna("").astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df

# Converts a product row into metadata dictionary for Chroma
def _to_metadata(row) -> Dict:
    return {
        "title": str(row["title"])[:240],
        "brand": row["brand"],
        "price": float(row["price"]) if pd.notna(row["price"]) else None,
        "avg_rating": float(row["average_rating"]) if pd.notna(row["average_rating"]) else None,
        "rating_n": int(row["rating_number"]) if pd.notna(row["rating_number"]) else None,
        "category": row["filtered_category"],
        "bucket": row["price_bucket"],
    }
    
# Generator to yield all indexed ASINs from ChromaDB in pages.
# Helps skip already embedded products.
def _iter_existing_ids(collection, page: int = DEFAULT_GET_PAGE) -> Iterable[str]:
    offset = 0
    while True:
        got = collection.get(limit=page, offset=offset)
        ids = got.get("ids") or []
        if not ids:
            break
        for _id in ids:
            yield _id
        if len(ids) < page:
            break
        offset += page

#Embed a list of texts with basic backoff on rate limiting (429/quota).Preserves input order.
def embed_batch(texts: List[str], model: str = EMBED_MODEL, max_retries: int = 5) -> List[List[float]]:
    client = _openai_client()
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower() or "quota" in msg.lower():
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise
    raise RuntimeError("Embedding failed after retries.")
    
# Adds new embeddings and their metadata to the Chroma collection.
def add_to_collection(collection, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]) -> None:
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)


# main indexer 
"""
Build/update the vector index:
    - Pages through MySQL (ORDER BY asin)
    - Skips ASINs already indexed in Chroma
    - Embeds 'search_text' in batches
    - Adds vectors + light metadata to Chroma

Returns (indexed_count, skipped_count, final_store_count).
"""
def index_all_products(
    sql_page: int = DEFAULT_SQL_PAGE,
    embed_batch_size: int = DEFAULT_EMBED_BATCH,
    get_page_size: int = DEFAULT_GET_PAGE,
    sleep_between_chunks: float = 0.2
) -> Tuple[int, int, int]:
    col = get_collection()

    total_rows = _count_total_products()
    pages = math.ceil(total_rows / sql_page)
    print(f"MySQL total rows: {total_rows:,}")
    print(f"Paging plan: {pages} pages * {sql_page} rows")
    print("OpenAI key present?", bool(OPENAI_API_KEY))

    # Build set of already indexed ids to skip on re-runs
    existing_ids = set(_iter_existing_ids(col, page=get_page_size))
    print("Already indexed in Chroma:", len(existing_ids))

    indexed = 0
    skipped = 0

    for p in range(pages):
        offset = p * sql_page
        df = _fetch_products_page(limit=sql_page, offset=offset)
        if df.empty:
            break

        mask_new = (~df["asin"].isin(existing_ids)) & (df["search_text"].str.len() > 0)
        df_new = df[mask_new].reset_index(drop=True)
        skipped += (len(df) - len(df_new))

        for i in range(0, len(df_new), embed_batch_size):
            chunk = df_new.iloc[i:i+embed_batch_size]
            if chunk.empty:
                continue

            vectors = embed_batch(chunk["search_text"].tolist())
            add_to_collection(
                col,
                ids=chunk["asin"].tolist(),
                embeddings=vectors,
                metadatas=[_to_metadata(r) for _, r in chunk.iterrows()]
            )
            indexed += len(chunk)
            if sleep_between_chunks:
                time.sleep(sleep_between_chunks)

        print(f"[page {p+1}/{pages}] added: {indexed:,} | skipped: {skipped:,} | store_count: {col.count():,}")

    final_count = col.count()
    print("\nDone.")
    print("Final Chroma count:", final_count)
    print("Newly indexed this run:", indexed)
    print("Skipped (already present or empty text):", skipped)

    return indexed, skipped, final_count

#quick semantic search (for testing/demo)
def embed_query(q: str) -> List[float]:
    client = _openai_client()
    return client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding


#Quick interactive search for notebooks.
#Returns a list of dicts with (asin, title, brand, price, avg_rating, rating_n, category, bucket, distance).

def semantic_search(query_text: str, k: int = 5) -> List[Dict]:
    col = get_collection()
    qv  = embed_query(query_text)
    res = col.query(query_embeddings=[qv], n_results=k)
    out = []
    ids    = res.get("ids", [[]])[0]
    metas  = res.get("metadatas", [[]])[0]
    dists  = res.get("distances", [[]])[0]
    for asin, md, dist in zip(ids, metas, dists):
        out.append({
            "asin": asin,
            "title": md.get("title"),
            "brand": md.get("brand"),
            "price": md.get("price"),
            "avg_rating": md.get("avg_rating"),
            "rating_n": md.get("rating_n"),
            "category": md.get("category"),
            "bucket": md.get("bucket"),
            "distance": float(dist),
        })
    return out