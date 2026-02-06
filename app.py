import streamlit as st
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from groq import Groq
import fitz  # PyMuPDF

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Resume Screening & Ranking",
    page_icon="🤖",
    layout="wide"
)

# ================= GLOBAL STYLES =================
st.markdown("""
<style>
body { background-color: #0e1117; }
.block-container { padding-top: 2rem; }
.card {
    background-color: #1e2228;
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}
.badge {
    padding: 0.3rem 0.7rem;
    border-radius: 8px;
    font-size: 13px;
    margin-right: 6px;
    display: inline-block;
}
.green { background-color: #0f5132; color: #00ffcc; }
.red { background-color: #842029; color: #ffb3b3; }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<h1 style='color:#00ffcc;'>🤖 AI Resume Screening & Ranking System</h1>
<p style='color:#9aa4b2; font-size:16px;'>
Explainable ATS powered by NLP & LLaMA-3
</p>
<hr>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.markdown("""
<h2 style='color:#00ffcc;'>🧠 ATS Control Panel</h2>
<p style='color:#9aa4b2; font-size:14px;'>
Upload job description & resumes
</p>
<hr>
""", unsafe_allow_html=True)

job_desc = st.sidebar.text_area(
    "📄 Job Description",
    height=180,
    placeholder="Paste the job description here..."
)

uploaded_files = st.sidebar.file_uploader(
    "📑 Upload Resume PDFs",
    type="pdf",
    accept_multiple_files=True
)

analyze_btn = st.sidebar.button("🚀 Analyze & Rank Resumes")

# ================= LOAD MODEL =================
@st.cache_resource(show_spinner="Loading NLP model...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ================= GROQ CLIENT =================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ================= SKILL MASTER LIST =================
tech_skills = [
    "python", "machine learning", "deep learning", "nlp", "sql",
    "tensorflow", "pytorch", "scikit-learn", "data science",
    "aws", "docker", "mlops", "pandas", "numpy", "power bi", "tableau"
]

# ================= HELPERS =================
def extract_text(file):
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return " ".join(page.get_text() for page in doc)

def extract_skills_from_jd(jd_text, skill_list):
    jd_lower = jd_text.lower()
    return list({skill for skill in skill_list if skill in jd_lower})

# ================= DASHBOARD =================
st.markdown("### 📊 Analysis Dashboard")

if not analyze_btn:
    st.info("⬅ Upload resumes and click **Analyze & Rank** to see results.")

# ================= MAIN LOGIC =================
if analyze_btn and uploaded_files and job_desc:

    # ---- Extract JD Skills ----
    jd_skills = extract_skills_from_jd(job_desc, tech_skills)

    st.markdown("### 📌 Skills Required (Extracted from JD)")
    st.write(", ".join(jd_skills) if jd_skills else "No skills detected")

    job_embedding = model.encode(job_desc, convert_to_tensor=True)

    results = []
    resume_texts = {}

    for file in uploaded_files:
        text = extract_text(file)
        resume_texts[file.name] = text
        text_lower = text.lower()

        resume_embedding = model.encode(text, convert_to_tensor=True)
        semantic_score = float(util.cos_sim(resume_embedding, job_embedding)) * 100

        matched = [s for s in jd_skills if s in text_lower]
        skill_score = (len(matched) / len(jd_skills)) * 100 if jd_skills else 0

        final_score = round((0.7 * semantic_score) + (0.3 * skill_score), 2)

        results.append({
            "Candidate": file.name,
            "Score": final_score,
            "Semantic %": round(semantic_score, 2),
            "Skill %": round(skill_score, 2),
            "Matched": matched,
            "Missing": list(set(jd_skills) - set(matched))
        })

    results = sorted(results, key=lambda x: x["Score"], reverse=True)

    # ================= TOP CANDIDATE =================
    st.success(f"🏆 Top Candidate: {results[0]['Candidate']} ({results[0]['Score']}%)")

    # ================= RESULT CARDS =================
    for r in results:
        st.markdown(f"""
        <div class="card">
            <h4>{r['Candidate']} — <span style='color:#00ffcc;'>{r['Score']}%</span></h4>
            <p>
                🔍 <b>Score Breakdown</b><br>
                Semantic Similarity: {r['Semantic %']}%<br>
                Skill Match: {r['Skill %']}%
            </p>
            <p><b>Matched Skills:</b><br>
            {" ".join([f"<span class='badge green'>{s}</span>" for s in r['Matched']]) or "None"}
            </p>
            <p><b>Missing Skills:</b><br>
            {" ".join([f"<span class='badge red'>{s}</span>" for s in r['Missing']]) or "None"}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ================= CSV DOWNLOAD =================
    df = pd.DataFrame([{
        "Candidate": r["Candidate"],
        "Final ATS %": r["Score"],
        "Semantic %": r["Semantic %"],
        "Skill Match %": r["Skill %"],
        "Matched Skills": ", ".join(r["Matched"]),
        "Missing Skills": ", ".join(r["Missing"])
    } for r in results])

    st.download_button(
        "⬇ Download ATS Ranking CSV",
        df.to_csv(index=False),
        "ats_ranked_candidates.csv",
        "text/csv"
    )

    # ================= LLM EXPLANATION =================
    top_candidate = results[0]["Candidate"]
    top_resume_text = resume_texts[top_candidate]

    prompt = f"""
You are a senior technical recruiter.

Job Description:
{job_desc}

Required Skills:
{jd_skills}

Candidate Resume:
{top_resume_text}

ATS Scoring Logic:
- 70% Semantic Similarity
- 30% Skill Match

Explain:
1. Why this candidate scored high or low
2. Missing critical skills
3. How the candidate can improve ATS score
4. Interview focus areas
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=700
    )

    st.subheader("🧠 LLaMA-3 Recruiter Explanation")
    st.write(completion.choices[0].message.content)

