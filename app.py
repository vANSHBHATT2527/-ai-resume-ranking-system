import streamlit as st
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from groq import Groq
import fitz  # PyMuPDF

st.set_page_config(page_title="AI Resume Ranking System", layout="wide")
st.title("🤖 AI Resume Screening & Ranking System (MNC Level)")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

tech_skills = [
    "python", "machine learning", "deep learning", "nlp", "sql", "tensorflow",
    "pytorch", "scikit-learn", "data science", "aws", "docker", "mlops",
    "pandas", "numpy", "power bi", "tableau"
]

def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return " ".join(page.get_text() for page in doc)

job_desc = st.text_area("📌 Paste Job Description")
uploaded_files = st.file_uploader("📂 Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("🚀 Analyze & Rank") and uploaded_files and job_desc:
    job_emb = model.encode(job_desc, convert_to_tensor=True)
    results = []

    for file in uploaded_files:
        text = extract_text(file)
        emb = model.encode(text, convert_to_tensor=True)
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
    st.dataframe(df)

    st.download_button("⬇ Download CSV", df.to_csv(index=False), "ranked_candidates.csv")

    top_resume = extract_text(uploaded_files[0])

    prompt = f"""
    You are a senior technical recruiter.
    Analyze this resume and job description.
    Resume: {top_resume}
    Job: {job_desc}
    Give improvement suggestions and interview focus areas.
    """

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600
    )

    st.subheader("🧠 LLaMA-3 Recruiter Feedback")
    st.write(completion.choices[0].message.content)
