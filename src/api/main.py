# This FastAPI app provides endpoints to:
# - Check system health (/health)
# - Recommend products (/recommend)
# - Explain recommendations using LLM (/explain)

from pathlib import Path
import sys
from typing import Optional, List, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Import core functions from retrieval module
# - main function to get product recommendations
# - generates LLM-based explanation
# - returns how many vectors are in Chroma DB
# - checks if MySQL is reachable

from src.embeddings.retrieval import (
    recommend,
    llm_explain,
    chroma_count, 
    mysql_ping,   
)

# FastAPI app setup
app = FastAPI(title="Smart Retail Assistant API", version="0.1.0")

# Open CORS for local dev / Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Models (Pydantic)

# Input model for product recommendation
class RecommendRequest(BaseModel):
    query: str
    price_min: Optional[float] = Field(None, ge=0)
    price_max: Optional[float] = Field(None, ge=0)
    category: Optional[Literal["Headphones", "Laptops"]] = None
    min_rating: Optional[float] = Field(None, ge=0, le=5)
    k_candidates: int = Field(120, ge=10, le=300)
    top_k: int = Field(8, ge=1, le=20)

# Output model for each product
class ProductOut(BaseModel):
    asin: str
    title: str
    brand: str
    price: float
    average_rating: float
    rating_number: int
    filtered_category: Literal["Headphones", "Laptops"]
    image_url: str
    price_bucket: Literal["budget", "midrange", "premium", "unknown"]

# Output model for /recommend endpoint
class RecommendResponse(BaseModel):
    products: List[ProductOut]

# Output model for /explain endpoint
class ExplainResponse(BaseModel):
    products: List[ProductOut]
    summary: str

# Routes 
#Health check endpoint.
# - Confirms app is running
# - Tests MySQL connectivity
# - Reports number of vectors in Chroma

@app.get("/health")
def health():
    return {
        "status": "ok",
        "db_ok": mysql_ping(),
        "chroma_count": chroma_count(),
    }

@app.post("/recommend", response_model=RecommendResponse)
def post_recommend(body: RecommendRequest):
    #Get recommendations
    df = recommend(
        query_text=body.query,
        price_min=body.price_min,
        price_max=body.price_max,
        category=body.category,
        min_rating=body.min_rating,
        k_candidates=body.k_candidates,
        top_k=body.top_k,
    )
    # Selects only the relevant fields to return
    products = df[[
        "asin", "title", "brand", "price", "average_rating",
        "rating_number", "filtered_category", "image_url", "price_bucket"
    ]].to_dict(orient="records")
    return {"products": products}

#Recommend products + provide an LLM-generated explanation.
#Useful for user-facing product summaries.

@app.post("/explain", response_model=ExplainResponse)
def post_explain(body: RecommendRequest):
    #Get recommendations
    df = recommend(
        query_text=body.query,
        price_min=body.price_min,
        price_max=body.price_max,
        category=body.category,
        min_rating=body.min_rating,
        k_candidates=body.k_candidates,
        top_k=body.top_k,
    )
     #Format product output
    products = df[[
        "asin", "title", "brand", "price", "average_rating",
        "rating_number", "filtered_category", "image_url", "price_bucket"
    ]].to_dict(orient="records")

    #Generate natural language explanation using LLM
    summary = llm_explain(body.query, df)
    return {"products": products, "summary": summary}
