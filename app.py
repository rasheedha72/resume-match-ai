import streamlit as st
import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Page settings
st.set_page_config(page_title="AI Resume Matcher", layout="centered")
st.title("🔍 AI Resume–Job Match Web App")
st.markdown("Upload your resume and paste a job description to see how well they match. 💼")

# Resume upload
resume_file = st.file_uploader("📄 Upload Your Resume (PDF only)", type=["pdf"])

# Job description input
job_description = st.text_area("📝 Paste the Job Description")

# Analyze button
if st.button("⚙️ Analyze Match"):

    if resume_file is not None and job_description.strip() != "":
        # Extract text from PDF
        with pdfplumber.open(resume_file) as pdf:
            resume_text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    resume_text += page_text

        if resume_text.strip() == "":
            st.error("⚠️ Could not extract text from the PDF. Please try another file.")
        else:
            # TF-IDF Matching
            documents = [resume_text, job_description]
            tfidf = TfidfVectorizer().fit_transform(documents)
            score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

            st.subheader("📊 Match Score")
            st.success(f"✅ Your resume matches **{round(score * 100, 2)}%** with the job description.")
    else:
        st.warning("⚠️ Please upload a resume and paste a job description.")
