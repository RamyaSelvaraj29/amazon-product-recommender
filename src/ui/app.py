# Smart Retail Assistant — Streamlit UI

import os, re, math, base64, pathlib, requests
import pandas as pd
import streamlit as st

#Backend API base URL( use env variables or default to localhost)
API_BASE = os.getenv("API_URL", "http://127.0.0.1:8000")

#Price ranges by category: (min, max, default)
PRICE_LIMITS = {
    "Headphones": (5, 500, 150),     # (min, max, default)
    "Laptops": (200, 4000, 1200),
    "Any": (1, 5000, 500)
}

# Streamlit app configuration
st.set_page_config(page_title="Smart Retail Assistant", page_icon="🛍️", layout="wide")

#Sets a full-page background image from a local file (JPG or PNG).Uses base64 encoding so it always works, even if deployed online.
def set_background():
    img_path = (pathlib.Path(__file__).parent / "bg.jpg")
    if not img_path.exists():
        img_path = (pathlib.Path(__file__).parent / "bg.png")
        if not img_path.exists():
            return  # no background, continue
    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    mime = "image/jpeg" if img_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:{mime};base64,{b64}");
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
        }}
        .block-container {{ padding-top: 1rem; padding-bottom: 2rem; }}
        [data-testid="stSidebar"] {{
            background: rgba(255, 255, 255, 0.88);
            border-right: 1px solid rgba(0,0,0,0.06);
        }}
        .card, .summary-card {{
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 12px;
            padding: 14px;
        }}
        .title {{ font-weight: 600; font-size: 0.95rem; line-height: 1.3rem; margin: 6px 0; }}
        .muted {{ color: #555; font-size: 0.85rem; }}
        .chip  {{ display:inline-block; padding: 2px 8px; background:#f2f2f2; border-radius:999px; font-size: 0.75rem; color:#333; }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

#Helper Functions 
#Sends POST request to FastAPI backend.
def call_api(path: str, payload: dict) -> dict:
    url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API error calling {url}: {e}")
        return {}

#Formats a number into currency (e.g., 100 -> $100.00)
def format_money(v):
    try:
        return f"${float(v):,.2f}"
    except:
        return "-"
#Displays star rating using HTML.
def star_html(rating: float) -> str:
    """Simple 5-star display."""
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return (
        "<span style='color:#f5a623;'>"
        + "&#9733;" * full
        + ("&#189;" if half else "")
        + f"<span style='color:#d9d9d9;'>{'&#9733;'*empty}</span></span>"
    )

# Product display format
def product_card(p: dict):
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        asin = str(p.get("asin","")).strip().upper()
        title = str(p.get("title","")).strip()
        brand = str(p.get("brand","")).strip()
        img = p.get("image_url") or ""
        url = f"https://www.amazon.com/dp/{asin}"  # may 404 if delisted; see disclaimer

        cols = st.columns([1, 3], vertical_alignment="center")
        with cols[0]:
            if img: st.image(img, use_container_width=True)
            else:   st.write("No image")

        with cols[1]:
            st.markdown(f"<div class='title'>{title}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='muted'>{brand} &nbsp;•&nbsp; {p['filtered_category']}</div>", unsafe_allow_html=True)
            c1, c2 = st.columns([1,1])
            with c1:
                st.markdown(f"**{format_money(p['price'])}**", unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f"{star_html(float(p['average_rating']))} &nbsp;"
                    f"<span class='muted'>{float(p['average_rating']):.1f} ({int(p['rating_number']):,})</span>",
                    unsafe_allow_html=True
                )
            st.markdown(f"<div class='chip'>{p['price_bucket'].title()}</div>", unsafe_allow_html=True)

            c3, c4 = st.columns([1,1])
            with c3:
                st.link_button("View on Amazon", url, use_container_width=True)
            with c4:
                # Compare shortlist toggle (persistent)
                key = f"cmp_{asin}"
                selected = st.checkbox("Compare", key=key, value=asin in st.session_state["compare"])
                if selected: st.session_state["compare"].add(asin)
                else:        st.session_state["compare"].discard(asin)

        st.markdown("</div>", unsafe_allow_html=True)


#Extract basic price hints from query:
# - under/below/<= 100 -> max=100
# - over/above/>= 300  -> min=300
# - between 300 and 600 or 300-600 -> min=300, max=600
def parse_price_from_query(text: str):
    q = text.lower()
    pmin = pmax = None

    # range forms
    m = re.search(r"(?:between|from)\s+\$?(\d+(?:\.\d{1,2})?)\s*(?:and|to|-)\s*\$?(\d+(?:\.\d{1,2})?)", q)
    if not m:
        m = re.search(r"\$?(\d+(?:\.\d{1,2})?)\s*[-–]\s*\$?(\d+(?:\.\d{1,2})?)", q)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        pmin, pmax = min(a,b), max(a,b)

    # upper-bound forms
    if pmax is None:
        m = re.search(r"(under|below|less than|up to|<=)\s+\$?(\d+(?:\.\d{1,2})?)", q)
        if m: pmax = float(m.group(2))

    # lower-bound forms
    if pmin is None:
        m = re.search(r"(over|above|more than|at least|>=)\s+\$?(\d+(?:\.\d{1,2})?)", q)
        if m: pmin = float(m.group(2))

    return pmin, pmax

# Avoid wiping on rerun by storing data in session
st.session_state.setdefault("last_results", [])
st.session_state.setdefault("last_summary", None)
st.session_state.setdefault("last_payload", None)
st.session_state.setdefault("compare", set())


# UI
st.markdown(
    """
    <h1 style='text-align:center; color:#DE7E5D; font-size:2.2rem; margin: 0.2em 0 0.6em;'>
        🛍️ Smart Retail Assistant
    </h1>
    """,
    unsafe_allow_html=True
)

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    category = st.selectbox("Category", ["Any", "Headphones", "Laptops"], index=0)
    lo, hi, default_max = PRICE_LIMITS["Any" if category == "Any" else category]
    no_budget = st.checkbox("No budget limit", value=False)
    price_max = None if no_budget else st.slider("Max price", int(lo), int(hi), int(default_max), step=1)
    min_rating = st.slider("Minimum rating", 1.0, 5.0, 4.0, step=0.1)
    top_k = st.select_slider("Results per page", options=[6, 8, 12], value=8)
    with st.expander("Advanced", expanded=False):
        k_candidates = st.slider("Semantic candidates (k)", 60, 300, 120, 10)
    explain_flag = st.checkbox("Generate AI summary", value=True)

# Query input + button
q = st.text_input("Describe what you want", placeholder="e.g., ANC over-ear headphones under $100")
run = st.button("Search", type="primary", use_container_width=True)
st.divider()

# Search: build payload (auto-apply price hints), call API, cache results
if run and q.strip():
    chosen_category = None if category == "Any" else category
    pmin, pmax = parse_price_from_query(q)
    # Apply detected bounds if present, otherwise use slider max
    chosen_min = float(int(pmin)) if pmin is not None else None
    chosen_max = float(int(pmax)) if pmax is not None else price_max

    payload = {
        "query": q.strip(),
        "price_min": chosen_min,
        "price_max": chosen_max,
        "category": chosen_category,
        "min_rating": float(min_rating),
        "k_candidates": int(k_candidates),
        "top_k": int(top_k),
    }

    res = call_api("recommend", payload)
    st.session_state["last_results"] = (res or {}).get("products", [])
    st.session_state["last_payload"] = payload

    if explain_flag and st.session_state["last_results"]:
        res2 = call_api("explain", payload)
        st.session_state["last_summary"] = (res2 or {}).get("summary")
    else:
        st.session_state["last_summary"] = None

# Always render from cached results
products = st.session_state["last_results"]

# Layout: results on left, AI summary on right
left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    if products:
        st.subheader(f"Results ({len(products)})")
        # draw grid (2 or 3 cols based on top_k)
        n_cols = 2 if top_k <= 8 else 3
        rows = math.ceil(len(products) / n_cols)
        i = 0
        for _ in range(rows):
            row_cols = st.columns(n_cols, gap="large")
            for c in row_cols:
                if i < len(products):
                    with c: product_card(products[i]); i += 1
        # Simple disclaimer about old ASINs
        st.markdown(
            "<div style='color:#d9534f; font-size:0.9rem;'>⚠️ Some product links may not work if items were delisted since 2023.</div>",
            unsafe_allow_html=True 
        )
        st.divider()
    elif run:
        st.info("No products matched your filters. Try adjusting price or rating.")

with right_col:
    st.subheader("AI Summary")
    st.markdown(
        f"<div class='summary-card'>{st.session_state['last_summary'] or '_No summary available._'}</div>",
        unsafe_allow_html=True
    )

# Compare section table
if st.session_state["compare"]:
    st.subheader("Compare")
    pool = {p["asin"]: p for p in products}  # last shown results
    rows = [pool[a] for a in list(st.session_state["compare"]) if a in pool]
    if rows:
        df_cmp = pd.DataFrame(rows)[
            ["asin","title","brand","price","average_rating","rating_number","price_bucket"]
        ]
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)
    else:
        st.caption("Add some items from results to compare.")

st.caption("Built with Streamlit • FastAPI • ChromaDB • OpenAI")
