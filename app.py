import streamlit as st
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from groq import Groq
import fitz  # PyMuPDF

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="AI Resume Screening & Ranking System",
    layout="wide"
)

st.title("🤖 AI Resume Screening & Ranking System (MNC Level)")

# -------------------- Load Model --------------------
@st.cache_resource(show_spinner="Loading NLP model...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------- Groq Client --------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------- Skills --------------------
tech_skills = [
    "python", "machine learning", "deep learning", "nlp", "sql",
    "tensorflow", "pytorch", "scikit-learn", "data science",
    "aws", "docker", "mlops", "pandas", "numpy", "power bi", "tableau"
]

# -------------------- PDF Text Extraction --------------------
def extract_text(file):
    file.seek(0)  # 🔑 CRITICAL FIX
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

# -------------------- UI Inputs --------------------
job_desc = st.text_area("📌 Paste Job Description", height=180)

uploaded_files = st.file_uploader(
    "📂 Upload Multiple Resume PDFs",
    type="pdf",
    accept_multiple_files=True
)

# -------------------- Main Logic --------------------
if st.button("🚀 Analyze & Rank") and uploaded_files and job_desc:

    job_embedding = model.encode(job_desc, convert_to_tensor=True)

    results = []
    resume_texts = {}  # store extracted text once

    for file in uploaded_files:
        text = extract_text(file)
        resume_texts[file.name] = text

        resume_embedding = model.encode(text, convert_to_tensor=True)
        score = float(util.cos_sim(resume_embedding, job_embedding)) * 100

        text_lower = text.lower()
        matched = [s for s in tech_skills if s in text_lower]
        missing = list(set(tech_skills) - set(matched))

        results.append({
            "Candidate": file.name,
            "Match %": round(score, 2),
            "Matched Skills": ", ".join(matched),
            "Missing Skills": ", ".join(missing)
        })

    # -------------------- Results Table --------------------
    df = pd.DataFrame(results).sort_values(by="Match %", ascending=False)

    st.subheader("🏆 Ranked Candidates")
    st.dataframe(df, use_container_width=True)

    # -------------------- Download CSV --------------------
    st.download_button(
        "⬇ Download Ranking CSV",
        df.to_csv(index=False),
        "ranked_candidates.csv",
        "text/csv"
    )

    # -------------------- LLM Feedback --------------------
    top_candidate = df.iloc[0]["Candidate"]
    top_resume_text = resume_texts[top_candidate]

    st.success(f"🥇 Top Candidate: {top_candidate}")

    prompt = f"""
You are a senior technical recruiter.

Job Description:
{job_desc}

Candidate Resume:
{top_resume_text}

Provide:
1. Why this candidate is suitable
2. Missing or weak technical skills
3. Interview focus areas
4. How to improve ATS score
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=700
    )

    st.subheader("🧠 LLaMA-3 Recruiter Feedback")
    st.write(completion.choices[0].message.content)
