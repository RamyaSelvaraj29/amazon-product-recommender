# Semantic product retrieval and explanation module
# - Embeds a user query with OpenAI
# - Finds similar products from Chroma vector DB
# - Filters results using SQL(price, rating, category)
# - Sorts and returns final top-k matches
# - Explains the results using OpenAI's GPT model

from typing import Optional, List, Literal
import pandas as pd
from sqlalchemy import create_engine, text
from openai import OpenAI
import chromadb
from config import OPENAI_API_KEY, MYSQL, PROJECT_ROOT

# OpenAI models
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL   = "gpt-4o-mini"

# Vector DB settings
COLLECTION_NAME = "products_2023_electronics"
VECTOR_DIR = PROJECT_ROOT / "vector_store"

# Database connection, Chroma, and OpenAI
URI = f"mysql+pymysql://{MYSQL['user']}:{MYSQL['password']}@{MYSQL['host']}:{MYSQL['port']}/{MYSQL['db']}"
engine = create_engine(URI, pool_pre_ping=True, future=True)
chromadb_client = chromadb.PersistentClient(path=str(VECTOR_DIR))
collection      = chromadb_client.get_or_create_collection(COLLECTION_NAME)
client = OpenAI(api_key=OPENAI_API_KEY)

# Health check functions
def mysql_ping() -> bool:
    """Return True if we can run SELECT 1 against MySQL."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

def chroma_count() -> int:
    """Return number of vectors currently in Chroma."""
    try:
        return collection.count()
    except Exception:
        return 0


# Embed a user query into a vector using OpenAI
def embed_query(q: str) -> List[float]:
    return client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
    
# Return top-k ASINs from Chroma based on semantic similarity
def chroma_candidates(query_text: str, k_candidates: int = 120) -> List[str]:
    qv = embed_query(query_text)
    res = collection.query(query_embeddings=[qv], n_results=k_candidates)
    return res["ids"][0] if res["ids"] else []

#Filter Chroma results with structured SQL logic
def sql_refine(
    asin_list: List[str],
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    category: Optional[Literal["Headphones","Laptops"]] = None,
    min_rating: Optional[float] = None
) -> pd.DataFrame:
    if not asin_list:
        return pd.DataFrame()
        
    # Adds filters dynamically
    placeholders = ",".join([f":a{i}" for i in range(len(asin_list))])
    params = {f"a{i}": a for i, a in enumerate(asin_list)}
    conds = []
    if price_min is not None:
        conds.append("price >= :pmin"); params["pmin"] = float(price_min)
    if price_max is not None:
        conds.append("price <= :pmax"); params["pmax"] = float(price_max)
    if category:
        conds.append("filtered_category = :cat"); params["cat"] = category
    if min_rating is not None:
        conds.append("average_rating >= :minr"); params["minr"] = float(min_rating)

    # Combine filters
    where_extra = (" AND " + " AND ".join(conds)) if conds else ""

    # Final SQL query 
    sql = text(f"""
        SELECT asin, title, brand, price, average_rating, rating_number,
               filtered_category, image_url, price_bucket
        FROM products
        WHERE asin IN ({placeholders}) {where_extra}
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params=params)

#Runs semantic search + Applies filters via SQL + sorts results and returns top-k.
def recommend(
    query_text: str,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    category: Optional[Literal["Headphones","Laptops"]] = None,
    min_rating: Optional[float] = None,
    k_candidates: int = 120,
    top_k: int = 8
) -> pd.DataFrame:
    # 1. Get candidated from Chroma
    asins = chroma_candidates(query_text, k_candidates)
    if not asins: return pd.DataFrame()

    #2. Filter with SQL
    df = sql_refine(asins, price_min=price_min, price_max=price_max, category=category, min_rating=min_rating)
    if df.empty: return df

    #3. Sort results by semantic rank and rating
    order = {a: i for i, a in enumerate(asins)}
    df["semantic_rank"] = df["asin"].map(order).fillna(1e9).astype(int)
    df = df.sort_values(
        ["semantic_rank", "average_rating", "rating_number"],
        ascending=[True, False, False]
    ).head(top_k).reset_index(drop=True)
    df["price"] = df["price"].astype(float)
    df["average_rating"] = df["average_rating"].astype(float)
    df["rating_number"] = df["rating_number"].astype(int)
    return df

# Generate LLM-based summary of the recommendations
def llm_explain(query: str, df: pd.DataFrame) -> str:
    if df.empty:
        return "I couldn't find matching products for your request."
    #Convert product rows to simple dictionaries
    items = [{
        "asin": r["asin"],
        "title": str(r["title"])[:140],
        "brand": r["brand"],
        "price": float(r["price"]),
        "rating": float(r["average_rating"]),
        "reviews": int(r["rating_number"]),
        "category": r["filtered_category"],
        "bucket": r["price_bucket"],
    } for _, r in df.iterrows()]

    #System instructions for the assistant
    sys_msg = ("You are a helpful retail assistant. Summarize products succinctly, "
               "compare trade-offs, and end with a one-line actionable suggestion. "
               "Be neutral and use US pricing.")

    #User message (with query and product list)
    user_msg = (f"User query: {query}\n\nTop products JSON:\n{items}\n\n"
                "Write 1 short paragraph (3–5 sentences), then 2–3 concise bullets (value pick, best overall, premium).")
    #Calls GPT model
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],
        temperature=0.3, max_tokens=280
    )
    return resp.choices[0].message.content.strip()
