import streamlit as st
import pdfplumber
import docx2txt
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import base64
import matplotlib.pyplot as plt
import json
import os
import spacy
import plotly.express as px
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB

from spacy.matcher import PhraseMatcher
from ui_components import (
    render_match_donut,
    render_resume_quality_gauge,
    render_skill_match_bars,
    candidate_card,
    render_leaderboard,
)
from analytics import compute_resume_quality, count_skill_occurrences, compute_overall_match

# Load spaCy model for skill extraction
nlp = spacy.load("en_core_web_sm")

# Pass the PhraseMatcher class (factory) to SkillExtractor so it can
# construct matchers itself. Passing an already-instantiated matcher
# caused skillNer to call the instance like a factory and raised
# "__call__() got an unexpected keyword argument 'attr'".
skill_extractor = None
skill_extractor_init_error = None
try:
    skill_extractor = SkillExtractor(nlp, skills_db=SKILL_DB, phraseMatcher=PhraseMatcher)
except Exception as _e:
    # Don't crash the whole app when skillNer fails to initialize.
    # Store the error for debugging and continue with degraded functionality.
    import traceback, sys
    skill_extractor_init_error = str(_e)
    traceback.print_exc(file=sys.stderr)

load_dotenv()

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama


# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="Resume Screening Agent", layout="wide")
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", ["Upload & Screening", "View Results", "Shortlisted Candidates"])
# Inject UI CSS for nicer visuals
from ui_components import inject_css
# Theme toggle (persisted in session_state)
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
with st.sidebar:
    # Radio keeps it simple and visible in the sidebar
    theme_choice = st.radio("Theme", ("Dark", "Light"), index=0 if st.session_state.theme == 'dark' else 1, key='theme_radio')
st.session_state.theme = theme_choice.lower()
inject_css(theme=st.session_state.theme)


# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

def get_resume_text(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    else:
        return ""


def compute_similarity(resume_text, jd_text):
    embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    jd_emb = embed.embed_query(jd_text)
    resume_emb = embed.embed_query(resume_text)

    sim = cosine_similarity([resume_emb], [jd_emb])[0][0]
    return round(float(sim) * 100, 2)


def extract_skills(text: str) -> List[str]:
    """Extract skills from text using skillNer"""
    try:
        # If skill_extractor failed to initialize, return empty skill list
        if skill_extractor is None:
            if skill_extractor_init_error:
                st.warning(f"skill_extractor not available: {skill_extractor_init_error}")
            return []

        # Process the text with skill extractor
        annotations = skill_extractor.annotate(text)
        
        # Extract unique skills and normalize them
        skills = set()
        for skill_type in ['full_matches', 'ngram_scored']:
            for skill in annotations['results'][skill_type]:
                skills.add(skill['doc_node_value'].lower())
                
        return list(skills)
    except Exception as e:
        st.warning(f"Skill extraction failed: {str(e)}")
        return []

def get_skill_matches(resume_skills: List[str], jd_skills: List[str]) -> Dict[str, Any]:
    """Compare skills between resume and job description"""
    # Convert to sets for easier comparison
    resume_skills_set = set(skill.lower() for skill in resume_skills)
    jd_skills_set = set(skill.lower() for skill in jd_skills)
    
    # Find matches and missing skills
    matches = list(resume_skills_set.intersection(jd_skills_set))
    missing = list(jd_skills_set - resume_skills_set)
    
    # Calculate match percentage
    match_percentage = (len(matches) / len(jd_skills_set)) * 100 if jd_skills_set else 0
    
    return {
        "present": sorted(matches),
        "missing": sorted(missing),
        "match_percentage": round(match_percentage, 2),
        "total_skills": len(resume_skills_set),
        "matched_skills": len(matches),
        "missing_skills": len(missing)
    }

def plot_skill_gaps(skills_analysis: Dict[str, Any]) -> None:
    """Generate a visualization of skill matches and gaps"""
    data = {
        'Category': ['Matched Skills', 'Missing Skills'],
        'Count': [
            skills_analysis['matched_skills'],
            skills_analysis['missing_skills']
        ]
    }
    
    fig = px.bar(
        data, 
        x='Category', 
        y='Count',
        color='Category',
        title=f"Skill Match: {skills_analysis['match_percentage']}%"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def analyze_candidate_with_skills(resume_text: str, jd_text: str) -> str:
    """
    Enhanced resume analysis with detailed skill extraction and matching
    """
    # First, extract skills using skillNer
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    
    # Get skill matches and gaps
    skills_analysis = get_skill_matches(resume_skills, jd_skills)
    
    # Use LLM for the rest of the analysis
    llm = ChatOllama(model="tinyllama")

    prompt = f"""
    Analyze the resume against the job description and provide a detailed analysis in a valid JSON format.
    
    JOB DESCRIPTION:
    {jd_text}
    
    RESUME:
    {resume_text}
    
    Return a JSON object with the following structure:
    {{
        "experience": {{
            "years": X,                            // Total years of experience
            "relevant_years": X,                   // Years relevant to the job
            "matches": ["role1 at Company A", ...] // Matching job titles/roles
        }},
        "education": {{
            "highest_degree": "Degree Name",      // Highest degree obtained
            "institution": "University Name",     
            "is_qualified": true/false,           // Meets education requirements
            "certifications": ["cert1", ...]      // Relevant certifications
        }},
        "keywords": ["keyword1", "keyword2", ...], // Important keywords from JD found in resume
        "summary": "Detailed analysis of the candidate's fit",
        "verdict": "Strong Match/Moderate Match/Not a Good Fit",
        "strengths": ["strength1", "strength2", ...],
        "weaknesses": ["weakness1", "weakness2", ...]
    }}
    
    IMPORTANT: Only return valid JSON. Do not include any other text before or after the JSON object.
    """

    try:
        response = llm.invoke(prompt)

        # Normalize the LLM output to a string. Different LLM wrappers may
        # return a string or an object with a `content` attribute.
        if hasattr(response, 'content'):
            json_str = str(response.content).strip()
        else:
            json_str = str(response).strip()

        # Strip markdown fences if present
        if json_str.startswith('```json'):
            json_str = json_str[7:-3].strip()
        elif json_str.startswith('```'):
            json_str = json_str[3:-3].strip()

        # If empty, treat as a failure and fall through to fallback handling
        if not json_str:
            raise ValueError("Empty LLM response")

        # Try to extract a JSON object from the response by finding the
        # first '{' and the last '}' and parsing that substring. This
        # handles cases where the model returns additional commentary.
        start = json_str.find('{')
        end = json_str.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = json_str[start:end+1]
        else:
            candidate = json_str

        try:
            analysis = json.loads(candidate)
        except Exception:
            # As a last attempt try to load the whole string
            analysis = json.loads(json_str)

        # Ensure we always include skill analysis
        analysis["skills"] = skills_analysis

        # Add skill-based strengths/weaknesses if not provided by LLM
        if not analysis.get('strengths') and skills_analysis['present']:
            analysis['strengths'] = [f"Strong in: {', '.join(skills_analysis['present'][:3])}..."]

        if not analysis.get('weaknesses') and skills_analysis['missing']:
            analysis['weaknesses'] = [f"Could improve in: {', '.join(skills_analysis['missing'][:3])}..."]

        return json.dumps(analysis)

    except Exception as e:
        # If JSON parsing or LLM invocation fails, return a stable fallback
        # JSON that includes the skill analysis and a short excerpt of the
        # raw LLM output for debugging.
        raw_excerpt = ''
        try:
            if 'json_str' in locals() and json_str:
                raw_excerpt = json_str[:500]
            elif hasattr(response, 'content'):
                raw_excerpt = str(response.content)[:500]
            else:
                raw_excerpt = str(response)[:500]
        except Exception:
            raw_excerpt = ''

        # Attempt a single reformatting/retry: ask the LLM to convert the
        # raw excerpt into valid JSON matching the expected schema. This
        # helps when the model returns prose or steps instead of JSON.
        try:
            reformat_prompt = (
                "You are a JSON formatter. Convert the following text into a valid JSON object"
                " that matches this schema:\n"
                "{\n  \"experience\":{...}, \n  \"education\":{...}, \n  \"keywords\":[], \n  \"summary\": \"...\", \n  \"verdict\": \"...\", \n  \"strengths\":[], \n  \"weaknesses\":[]\n}\n"
                "ONLY return the JSON object and nothing else.\n\n"
                "Here is the text to convert:\n\n" + raw_excerpt
            )

            re_resp = llm.invoke(reformat_prompt)
            if hasattr(re_resp, 'content'):
                re_str = str(re_resp.content).strip()
            else:
                re_str = str(re_resp).strip()

            if re_str.startswith('```json'):
                re_str = re_str[7:-3].strip()
            elif re_str.startswith('```'):
                re_str = re_str[3:-3].strip()

            # Try parsing the reformatted output
            start_r = re_str.find('{')
            end_r = re_str.rfind('}')
            candidate_r = re_str[start_r:end_r+1] if (start_r != -1 and end_r != -1 and end_r>start_r) else re_str
            parsed = json.loads(candidate_r)
            # Merge skill analysis and return
            parsed['skills'] = skills_analysis
            # Mark that we used a reformat attempt
            parsed['llm_retries'] = 1
            parsed['llm_raw'] = raw_excerpt
            return json.dumps(parsed)
        except Exception:
            # Final fallback if reformat also fails
            fallback = {
                "skills": skills_analysis,
                "experience": {"years": 0, "relevant_years": 0, "matches": []},
                "education": {"highest_degree": "", "institution": "", "is_qualified": False, "certifications": []},
                "keywords": [],
                "summary": f"AI analysis failed, but skill extraction worked. {str(e)}",
                "llm_raw": raw_excerpt,
                "llm_retries": 1,
                "verdict": "Partial Analysis",
                "strengths": [f"Strong in: {', '.join(skills_analysis['present'][:3])}..."] if skills_analysis['present'] else [],
                "weaknesses": [f"Could improve in: {', '.join(skills_analysis['missing'][:3])}..."] if skills_analysis['missing'] else []
            }

            return json.dumps(fallback)


def generate_bar_chart(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df['Filename'], df['Score'])
    plt.xticks(rotation=45, ha="right")
    plt.title("Candidate Score Comparison")
    plt.ylabel("Match Percentage")
    plt.tight_layout()
    return fig


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def safe_json_loads(json_string):
    try:
        # Find the start and end of the JSON object
        start_index = json_string.find('{')
        end_index = json_string.rfind('}') + 1
        if start_index != -1 and end_index != 0:
            json_part = json_string[start_index:end_index]
            return json.loads(json_part)
    except (json.JSONDecodeError, IndexError):
        pass
    # Return a default structure if parsing fails
    return {
        "skills_present": [],
        "skills_missing": [],
        "summary": "AI analysis failed.",
        "verdict": "Error"
    }


# ----------------------------------------------------
# SESSION STATE STORAGE
# ----------------------------------------------------
if "results" not in st.session_state:
    st.session_state["results"] = []

if "shortlisted" not in st.session_state:
    st.session_state["shortlisted"] = []


# ----------------------------------------------------
# PAGE 1 — Upload & Screening
# ----------------------------------------------------
if page == "Upload & Screening":

    st.title("🧑‍💼 Resume Screening Agent")
    st.write("Upload resumes + job description → AI screens & ranks candidates.")

    jd = st.text_area("📌 Paste Job Description")

    resume_files = st.file_uploader(
        "📂 Upload Multiple Resumes (PDF/DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if st.button("🔍 Start Screening"):

        if not jd or not resume_files:
            st.error("Please enter JD & upload resumes.")
            st.stop()

        results = []

        for file in resume_files:
            with st.spinner(f"Processing {file.name}..."):
                text = get_resume_text(file)
                score = compute_similarity(text, jd)
                ai_json = analyze_candidate_with_skills(text, jd)
                st.write("RAW AI JSON:", ai_json)

            results.append({
                "Filename": file.name,
                "Score": score,
                "AI_Analysis": ai_json
            })

        results = sorted(results, key=lambda x: x["Score"], reverse=True)

        # persist results and JD for the dashboard
        st.session_state.results = results
        st.session_state.last_jd = jd
        st.success("Screening Completed! Go to 'View Results' tab.")


# ----------------------------------------------------
# PAGE 2 — View Results
# ----------------------------------------------------
elif page == "View Results":

    # Header
    st.markdown("<div class='top-title'>📊 Ranked Results</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Review ranked candidates, explore skill matches and generate interview guides.</div>", unsafe_allow_html=True)

    if len(st.session_state.results) == 0:
        st.warning("No results yet. Go to 'Upload & Screening'")
        st.stop()

    results = st.session_state.results

    # Table view
    data = {
        "Filename": [r["Filename"] for r in results],
        "Score": [r["Score"] for r in results],
        # Use .get to avoid KeyError if 'verdict' is missing in LLM output
        "Verdict": [safe_json_loads(r["AI_Analysis"]).get("verdict", "No Verdict") for r in results]
    }

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # Bar chart
    st.subheader("📈 Score Comparison")
    fig = generate_bar_chart(df)
    st.pyplot(fig)

    # CSV download
    st.subheader("⬇ Download Results")
    csv = convert_df_to_csv(df)
    st.download_button(
        "Download Ranked CSV",
        csv,
        "ranked_results.csv",
        "text/csv",
        key="download-csv"
    )

    # Show Detailed AI Analysis
    st.subheader("📋 Detailed Candidate Analysis")

    for r in results:
        analysis = safe_json_loads(r["AI_Analysis"])
        
        with st.expander(f"🔍 {r['Filename']} — {r['Score']}% — {analysis.get('verdict', 'No Verdict')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Score and Verdict
                st.metric("Match Score", f"{r['Score']}%")
                
                # Skills Section with Visualization
                if 'skills' in analysis:
                    st.subheader("🛠️ Skills Analysis")
                    
                    # Show skill match visualization
                    plot_skill_gaps(analysis['skills'])
                    
                    # Show matched skills in expandable section
                    if analysis['skills'].get('present'):
                        with st.expander(f"✅ Matching Skills ({len(analysis['skills']['present'])})"):
                            st.write(", ".join([f"`{skill}`" for skill in analysis['skills']['present']]))
                    
                    # Show missing skills in expandable section
                    if analysis['skills'].get('missing'):
                        with st.expander(f"❌ Missing Skills ({len(analysis['skills']['missing'])})"):
                            st.write(", ".join([f"`{skill}`" for skill in analysis['skills']['missing']]))
                    
                    # Show skill statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Match Percentage", f"{analysis['skills'].get('match_percentage', 0)}%")
                    with col2:
                        st.metric("Skills Matched", analysis['skills'].get('matched_skills', 0))
                    with col3:
                        st.metric("Skills Missing", analysis['skills'].get('missing_skills', 0))
                
                # Education
                if 'education' in analysis and analysis['education'].get('highest_degree'):
                    st.subheader("🎓 Education")
                    st.write(f"**{analysis['education'].get('highest_degree', 'N/A')}**")
                    st.write(analysis['education'].get('institution', 'N/A'))
                    
                    if analysis['education'].get('certifications'):
                        st.write("**Certifications:**")
                        st.write(", ".join(analysis['education']['certifications']))
            
            with col2:
                # Experience
                if 'experience' in analysis:
                    st.subheader("💼 Experience")
                    st.write(f"**Total Experience:** {analysis['experience'].get('years', 0)} years")
                    st.write(f"**Relevant Experience:** {analysis['experience'].get('relevant_years', 0)} years")
                    
                    if analysis['experience'].get('matches'):
                        st.write("**Matching Roles:**")
                        for match in analysis['experience']['matches']:
                            st.write(f"- {match}")
                
                # Keywords
                if 'keywords' in analysis and analysis['keywords']:
                    st.subheader("🔑 Keywords Matched")
                    st.write(" ".join([f"`{kw}`" for kw in analysis['keywords']]))
            
            # Summary and Verdict
            st.subheader("📝 Summary")
            st.write(analysis.get('summary', 'No summary available.'))
            
            # Strengths and Weaknesses
            col3, col4 = st.columns(2)
            with col3:
                if 'strengths' in analysis and analysis['strengths']:
                    st.subheader("✅ Strengths")
                    for strength in analysis['strengths']:
                        st.write(f"- {strength}")
            
            with col4:
                if 'weaknesses' in analysis and analysis['weaknesses']:
                    st.subheader("⚠️ Areas for Improvement")
                    for weakness in analysis['weaknesses']:
                        st.write(f"- {weakness}")
            
            # Shortlist button at the bottom
            if st.button(f"⭐ Shortlist {r['Filename']}", key=f"shortlist_{r['Filename']}"):
                if r not in st.session_state.shortlisted:
                    st.session_state.shortlisted.append(r)
                    st.success(f"{r['Filename']} has been added to shortlisted candidates!")
                else:
                    st.warning("This candidate is already in your shortlist!")
            
            st.write("---")


# ----------------------------------------------------
# PAGE 3 — Shortlisted Candidates
# ----------------------------------------------------
elif page == "Shortlisted Candidates":
    st.title("🏆 Shortlisted Candidates")
    
    if len(st.session_state.shortlisted) == 0:
        st.info("No candidates have been shortlisted yet. Go to 'View Results' to shortlist candidates.")
        st.stop()
    
    # Add option to clear all shortlisted candidates
    if st.button("🗑️ Clear All Shortlisted Candidates", type="primary"):
        st.session_state.shortlisted = []
        st.experimental_rerun()
    
    st.write(f"### 📋 {len(st.session_state.shortlisted)} candidates shortlisted")
    
    # Create tabs for each shortlisted candidate
    tabs = st.tabs([f"{i+1}. {cand['Filename']} ({cand['Score']}%)" 
                   for i, cand in enumerate(st.session_state.shortlisted)])
    
    for idx, tab in enumerate(tabs):
        with tab:
            cand = st.session_state.shortlisted[idx]
            analysis = safe_json_loads(cand["AI_Analysis"])
            
            # Header with action buttons
            col1, col2 = st.columns([4, 1])
            with col1:
                st.subheader(f"{cand['Filename']}")
            with col2:
                if st.button("❌ Remove", key=f"remove_{cand['Filename']}"):
                    st.session_state.shortlisted.remove(cand)
                    st.experimental_rerun()
            
            # Main content
            col1, col2 = st.columns(2)
            
            with col1:
                # Score and Verdict
                st.metric("Match Score", f"{cand['Score']}%")
                
                # Skills Section with Visualization
                if 'skills' in analysis:
                    st.subheader("🛠️ Skills Analysis")
                    
                    # Show skill match visualization
                    plot_skill_gaps(analysis['skills'])
                    
                    # Show matched skills in expandable section
                    if analysis['skills'].get('present'):
                        with st.expander(f"✅ Matching Skills ({len(analysis['skills']['present'])})"):
                            st.write(", ".join([f"`{skill}`" for skill in analysis['skills']['present']]))
                    
                    # Show missing skills in expandable section
                    if analysis['skills'].get('missing'):
                        with st.expander(f"❌ Missing Skills ({len(analysis['skills']['missing'])})"):
                            st.write(", ".join([f"`{skill}`" for skill in analysis['skills']['missing']]))
                    
                    # Show skill statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Match Percentage", f"{analysis['skills'].get('match_percentage', 0)}%")
                    with col2:
                        st.metric("Skills Matched", analysis['skills'].get('matched_skills', 0))
                    with col3:
                        st.metric("Skills Missing", analysis['skills'].get('missing_skills', 0))
                
                # Education
                if 'education' in analysis and analysis['education'].get('highest_degree'):
                    st.subheader("🎓 Education")
                    st.write(f"**{analysis['education'].get('highest_degree', 'N/A')}**")
                    st.write(analysis['education'].get('institution', 'N/A'))
                    
                    if analysis['education'].get('certifications'):
                        st.write("**Certifications:**")
                        st.write(", ".join(analysis['education']['certifications']))
            
            with col2:
                # Experience
                if 'experience' in analysis:
                    st.subheader("💼 Experience")
                    st.write(f"**Total Experience:** {analysis['experience'].get('years', 0)} years")
                    st.write(f"**Relevant Experience:** {analysis['experience'].get('relevant_years', 0)} years")
                    
                    if analysis['experience'].get('matches'):
                        st.write("**Matching Roles:**")
                        for match in analysis['experience']['matches']:
                            st.write(f"- {match}")
                
                # Keywords
                if 'keywords' in analysis and analysis['keywords']:
                    st.subheader("🔑 Keywords Matched")
                    st.write(" ".join([f"`{kw}`" for kw in analysis['keywords']]))
            
            # Summary and Verdict
            st.subheader("📝 Summary")
            st.write(analysis.get('summary', 'No summary available.'))
            
            # Strengths and Weaknesses
            col3, col4 = st.columns(2)
            with col3:
                if 'strengths' in analysis and analysis['strengths']:
                    st.subheader("✅ Strengths")
                    for strength in analysis['strengths']:
                        st.write(f"- {strength}")
            
            with col4:
                if 'weaknesses' in analysis and analysis['weaknesses']:
                    st.subheader("⚠️ Areas for Improvement")
                    for weakness in analysis['weaknesses']:
                        st.write(f"- {weakness}")
            
            st.write("---")

