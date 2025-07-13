import streamlit as st
import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# List of common tech skills
common_skills = [
    "python", "java", "c++", "c", "html", "css", "javascript",
    "machine learning", "deep learning", "nlp", "data science", 
    "computer vision", "pandas", "numpy", "matplotlib", "seaborn",
    "tensorflow", "keras", "pytorch", "sql", "mongodb", "git", 
    "github", "streamlit", "flask", "fastapi", "scikit-learn", 
    "power bi", "excel", "tableau"
]

# Streamlit page config
st.set_page_config(page_title="AI Resume Matcher", layout="centered")
st.title("🔍 AI Resume–Job Match Web App")
st.markdown("Upload your resume and paste a job description to see how well they match. 💼")

# Upload resume
resume_file = st.file_uploader("📄 Upload Your Resume (PDF only)", type=["pdf"])

# Job description input
job_description = st.text_area("📝 Paste the Job Description")

# Main button
if st.button("⚙️ Analyze Match"):

    if resume_file is not None and job_description.strip() != "":
        # Extract text from PDF
        with pdfplumber.open(resume_file) as pdf:
            resume_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    resume_text += text

        if resume_text.strip() == "":
            st.error("⚠️ Could not extract text from the resume. Try another file.")
        else:
            # TF-IDF Score
            documents = [resume_text, job_description]
            tfidf = TfidfVectorizer().fit_transform(documents)
            score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            match_percent = round(score * 100, 2)

            # Display Match Score
            st.subheader("📊 Match Score")
            st.progress(match_percent / 100)  # Show progress bar
            st.write(f"**{match_percent}% Match** 🎯")

            # Skill Analysis
            resume_text_lower = resume_text.lower()
            jd_text_lower = job_description.lower()

            matched_skills = [skill for skill in common_skills if skill in resume_text_lower and skill in jd_text_lower]
            missing_skills = [skill for skill in common_skills if skill in jd_text_lower and skill not in resume_text_lower]

            st.markdown("---")
            st.subheader("📌 Skills Breakdown")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("✅ **Matched Skills**")
                if matched_skills:
                    st.success(", ".join(matched_skills))
                else:
                    st.warning("No skills matched.")

            with col2:
                st.markdown("❌ **Missing Skills**")
                if missing_skills:
                    st.error(", ".join(missing_skills))
                else:
                    st.success("All key skills are present!")

            st.markdown("---")
            st.caption("Built with 💖 by Rasheedha | Powered by Python + Streamlit + NLP")

    else:
        st.warning("⚠️ Please upload a resume and paste a job description.")
