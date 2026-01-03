import os
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai

st.set_page_config(page_title="SQL Schema Finder", page_icon="🔎")

st.title("🔎 Where does this data live?")
st.write("Ask questions like: *Where is A&E attendance stored?* or *Which table has referral date?*")

# --- Load metadata ---
@st.cache_data
def load_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic cleanup
    for col in ["schema_name", "table_name", "column_name", "data_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")
    if "notes" not in df.columns:
        df["notes"] = ""
    return df

df = load_metadata("schema_metadata.csv")

# Build a searchable text field for each row
df["search_text"] = (
    df["schema_name"] + "." + df["table_name"] + " " +
    df["column_name"] + " " + df["data_type"] + " " +
    df["notes"].fillna("")
)

@st.cache_resource
def build_search_index(texts):
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(texts)
    return vec, X

vectorizer, X = build_search_index(df["search_text"].tolist())

def top_matches(question: str, k: int = 15) -> pd.DataFrame:
    qv = vectorizer.transform([question])
    sims = cosine_similarity(qv, X).flatten()
    top_idx = sims.argsort()[::-1][:k]
    out = df.iloc[top_idx].copy()
    out["score"] = sims[top_idx]
    return out

# --- Gemini setup ---
api_key = st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    st.info("Admin setup: add GEMINI_API_KEY in Streamlit Secrets to enable full chatbot answers.")
else:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

example_prompts = [
    "Where would I find A&E attendances?",
    "Which table contains referral date?",
    "Is there a table for outpatient appointments?",
    "What column might contain discharge date?"
]

with st.expander("Example questions"):
    for p in example_prompts:
        st.write("• " + p)

question = st.text_input("Your question", placeholder="e.g. Where is inpatient discharge date stored?")

if question:
    matches = top_matches(question, k=15)

    st.subheader("Closest matches in your schema metadata")
    st.dataframe(
        matches[["schema_name", "table_name", "column_name", "data_type", "notes", "score"]],
        use_container_width=True
    )

    if api_key:
        # Build a compact context for the LLM
        context_rows = []
        for _, r in matches.iterrows():
            context_rows.append(
                f"{r['schema_name']}.{r['table_name']}.{r['column_name']} ({r['data_type']}) - {r.get('notes','')}"
            )
        context = "\n".join(context_rows)

        prompt = f"""
You are a data catalogue assistant. The user will ask where data lives.
Answer using ONLY the schema metadata below. If unsure, say you are not certain and suggest what to check next.

User question:
{question}

Schema metadata (top matches):
{context}

Return:
1) Best guess table(s) and column(s)
2) Why you think that
3) A short follow-up question to confirm meaning (if needed)
"""

        with st.spinner("Thinking..."):
            resp = model.generate_content(prompt)

        st.subheader("Chatbot answer")
        st.write(resp.text)

