import streamlit as st
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from groq import Groq

# Safe import for PyMuPDF
try:
    import fitz  # PyMuPDF
except Exception as e:
    st.error("PyMuPDF (fitz) is not installed correctly. Please check requirements.txt.")
    st.stop()

# Load SBERT model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

sbert = load_model()

# Load Groq client securely from Streamlit Secrets
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Tech skill list
tech_skills = [
    "python", "machine learning", "deep learning", "nlp", "sql", "tensorflow",
    "pytorch", "scikit-learn", "data science", "aws", "docker", "mlops",
    "pandas", "numpy", "power bi", "tableau"
]

# PDF text extraction
def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# UI
st.set_page_config(page_title="AI Resume Ranking System", layout="wide")
st.title("🤖 AI Resume Screening & Ranking System (MNC Level)")

job_desc = st.text_area("📌 Paste Job Description", height=200)
uploaded_files = st.file_uploader("📂 Upload Multiple Resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("🚀 Analyze & Rank") and uploaded_files and job_desc:
    results = []
    job_emb = sbert.encode(job_desc, convert_to_tensor=True)

    for file in uploaded_files:
        text = extract_text(file)
        emb = sbert.encode(text, convert_to_tensor=True)
        score = float(util.cos_sim(emb, job_emb)) * 100

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

    st.subheader("🏆 Ranked Candidates")
    st.dataframe(df, use_container_width=True)

    # CSV Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Ranking CSV", csv, "ranked_candidates.csv", "text/csv")

    top_candidate = df.iloc[0]["Candidate"]
    st.success(f"🥇 Top Candidate: {top_candidate}")

    # LLaMA-3 Feedback
    top_resume_text = extract_text(uploaded_files[0])

    prompt = f"""
    You are a senior technical recruiter.

    Job Description:
    {job_desc}

    Resume:
    {top_resume_text}

    Provide:
    1. Why this candidate is a good fit
    2. Missing technical skills
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
