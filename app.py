import streamlit as st
import os
import pandas as pd
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import fitz  # PyMuPDF

st.set_page_config(page_title="AI Resume Ranking System", layout="wide")
st.title("🤖 AI Resume Screening & Ranking System (MNC Level)")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model

tokenizer, model = load_model()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

tech_skills = [
    "python", "machine learning", "deep learning", "nlp", "sql", "tensorflow",
    "pytorch", "scikit-learn", "data science", "aws", "docker", "mlops",
    "pandas", "numpy", "power bi", "tableau"
]

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return " ".join(page.get_text() for page in doc)

job_desc = st.text_area("📌 Paste Job Description")
uploaded_files = st.file_uploader("📂 Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("🚀 Analyze & Rank") and uploaded_files and job_desc:
    job_emb = embed_text(job_desc).numpy()
    results = []

    for file in uploaded_files:
        text = extract_text(file)
        emb = embed_text(text).numpy()
        score = cosine_similarity(emb, job_emb)[0][0] * 100

        text_lower = text.lower()
        matched = [s for s in tech_skills if s in text_lower]
        missing = list(set(tech_skills) - set(matched))

        results.append({
            "Candidate": file.name,
            "Match %": round(score, 2),
            "Matched Skills": ", ".join(matched),
            "Missing Skills": ", ".join(missing)
        })

    df = pd.DataFrame(results).sort_values(by="Match %", ascending=False)
    st.dataframe(df)

    st.download_button("⬇ Download CSV", df.to_csv(index=False), "ranked_candidates.csv")

    top_resume = extract_text(uploaded_files[0])
    prompt = f"""
    You are a technical recruiter.
    Analyze this resume and JD and give improvement tips:
    Resume: {top_resume}
    JD: {job_desc}
    """

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600
    )

    st.subheader("🧠 LLaMA-3 Feedback")
    st.write(completion.choices[0].message.content)
