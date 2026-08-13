import io
import json
import logging
import re
from collections import Counter
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from crewai import Crew, LLM, Process, Task

from CrewaiAgents.CoverLetterAgent import CoverLetterAgent
from CrewaiAgents.JDUnderstandingAgent import JDUnderstandingAgent
from CrewaiAgents.MatchingAgent import MatchingAgent
from CrewaiAgents.ResumeEnhancerAgent import ResumeEnhancerAgent
from CrewaiAgents.ResumeParsingAgent import ResumeParsingAgent

load_dotenv()

try:
    import graphviz

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "of", "on", "or", "our", "that", "the", "this", "to",
    "we", "will", "with", "you", "your", "years", "work", "role", "job",
    "candidate", "experience", "required", "preferred", "including", "using",
}


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {background: linear-gradient(145deg, #f8fafc 0%, #eef2ff 100%);}
        [data-testid="stHeader"] {background: transparent;}
        .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
        .hero {
            padding: 2.2rem; border-radius: 22px; color: white;
            background: linear-gradient(120deg, #111827 0%, #312e81 55%, #4f46e5 100%);
            box-shadow: 0 18px 45px rgba(49, 46, 129, .20); margin-bottom: 1.4rem;
        }
        .hero h1 {font-size: 2.25rem; margin: 0 0 .55rem; letter-spacing: -.03em;}
        .hero p {font-size: 1.05rem; color: #e0e7ff; margin: 0; max-width: 760px;}
        .eyebrow {font-size: .78rem; letter-spacing: .13em; text-transform: uppercase;
                  color: #c7d2fe; font-weight: 700; margin-bottom: .55rem;}
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,.9); border: 1px solid #e2e8f0;
            border-radius: 18px; box-shadow: 0 8px 24px rgba(15,23,42,.05);
        }
        div.stButton > button {border-radius: 10px; font-weight: 650; min-height: 2.8rem;}
        div[data-testid="stMetric"] {background: white; border: 1px solid #e2e8f0;
            padding: 1rem; border-radius: 14px;}
        .status-ready {padding: .7rem 1rem; border-radius: 10px; background: #ecfdf5;
                       color: #047857; border: 1px solid #a7f3d0;}
        .small-note {color: #64748b; font-size: .88rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def extract_text_from_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as exc:
        logger.exception("PDF extraction failed")
        raise ValueError("The PDF could not be read. Try exporting it again or paste the text.") from exc


def normalize_tokens(text):
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-/]{1,}", text.lower())


def ats_keyword_analysis(resume_text, jd_text, limit=24):
    """Deterministic, explainable keyword coverage; not an LLM-generated score."""
    resume_tokens = set(normalize_tokens(resume_text))
    jd_counts = Counter(
        token for token in normalize_tokens(jd_text)
        if token not in STOP_WORDS and len(token) > 2
    )
    ranked = [token for token, _ in jd_counts.most_common(limit)]
    matched = [token for token in ranked if token in resume_tokens]
    missing = [token for token in ranked if token not in resume_tokens]
    score = round((len(matched) / len(ranked)) * 100) if ranked else 0
    return {"score": score, "matched": matched, "missing": missing, "keywords": ranked}


def clean_result(result, preferred_key):
    raw = getattr(result, "raw", str(result))
    try:
        parsed = json.loads(raw)
        return parsed.get(preferred_key, raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def build_report(resume_name, ats, results):
    sections = [
        "# AI Resume–Job Description Analysis",
        f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        f"Resume: {resume_name or 'Pasted resume text'}",
        "",
        "## ATS Keyword Coverage",
        f"Coverage score: **{ats['score']}%**",
        f"Matched keywords: {', '.join(ats['matched']) or 'None identified'}",
        f"Missing keywords: {', '.join(ats['missing']) or 'None identified'}",
    ]
    labels = {
        "match": "AI Match Analysis",
        "enhancement": "Resume Recommendations",
        "cover_letter": "Tailored Cover Letter",
    }
    for key, label in labels.items():
        if results.get(key):
            sections.extend(["", f"## {label}", results[key]])
    sections.extend([
        "",
        "---",
        "Note: Keyword coverage is a text-overlap indicator, not a guarantee of ATS ranking or employment outcome.",
    ])
    return "\n\n".join(sections)


def initialize_pipeline(resume_text, jd_text):
    llm = LLM(model="ollama/llama3:instruct", base_url="http://localhost:11434")
    agents = {
        "resume": ResumeParsingAgent(llm=llm),
        "jd": JDUnderstandingAgent(llm=llm),
        "matcher": MatchingAgent(llm=llm),
        "enhancer": ResumeEnhancerAgent(llm=llm),
        "cover": CoverLetterAgent(llm=llm),
    }
    tasks = {
        "resume": Task(description=resume_text, expected_output="Structured resume data", agent=agents["resume"]),
        "jd": Task(description=jd_text, expected_output="Structured job-description data", agent=agents["jd"]),
    }
    tasks["match"] = Task(
        description="Match the resume to the job description.",
        expected_output="A clear match score and evidence-based analysis.",
        agent=agents["matcher"], context=[tasks["resume"], tasks["jd"]],
    )
    tasks["enhancement"] = Task(
        description="Recommend truthful resume improvements based on the job description.",
        expected_output="Prioritized, actionable resume recommendations. Never invent qualifications.",
        agent=agents["enhancer"], context=[tasks["resume"], tasks["jd"]],
    )
    tasks["cover_letter"] = Task(
        description="Generate a tailored cover letter using only facts found in the resume.",
        expected_output="A concise professional cover letter.",
        agent=agents["cover"], context=[tasks["resume"], tasks["jd"]],
    )
    return agents, tasks


def run_workflow(kind, resume_text, jd_text):
    agents, tasks = initialize_pipeline(resume_text, jd_text)
    config = {
        "match": ([agents["resume"], agents["jd"], agents["matcher"]], [tasks["resume"], tasks["jd"], tasks["match"]], "match_summary"),
        "enhancement": ([agents["resume"], agents["jd"], agents["enhancer"]], [tasks["resume"], tasks["jd"], tasks["enhancement"]], "resume_enhancement"),
        "cover_letter": ([agents["resume"], agents["jd"], agents["cover"]], [tasks["resume"], tasks["jd"], tasks["cover_letter"]], "cover_letter"),
    }
    selected_agents, selected_tasks, key = config[kind]
    crew = Crew(
        agents=selected_agents, tasks=selected_tasks, process=Process.sequential,
        verbose=False, name=f"Resume JD {kind.title()} Crew",
    )
    return clean_result(crew.kickoff(), key)


st.set_page_config(page_title="CareerFit AI", page_icon="✦", layout="wide")
inject_styles()

if "results" not in st.session_state:
    st.session_state.results = {}

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Multi-agent career intelligence</div>
      <h1>CareerFit AI</h1>
      <p>Compare a resume with a job description, uncover ATS keyword gaps, improve positioning, and draft a tailored cover letter.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("How it works")
    st.caption("1 · Add your resume")
    st.caption("2 · Add the job description")
    st.caption("3 · Review ATS coverage")
    st.caption("4 · Run an AI workflow")
    st.divider()
    st.info("Runs locally with Ollama. Make sure Ollama is active and the configured model is installed.")
    with st.expander("Agent workflow"):
        if GRAPHVIZ_AVAILABLE:
            dot = graphviz.Digraph()
            dot.attr(rankdir="TB")
            dot.node("I", "Resume + Job Description")
            dot.node("P", "Parser Agents")
            dot.node("A", "Matcher / Enhancer / Writer")
            dot.node("O", "Application Toolkit")
            dot.edges([("I", "P"), ("P", "A"), ("A", "O")])
            st.graphviz_chart(dot, use_container_width=True)
        else:
            st.caption("Resume + JD → Parser agents → Specialist agent → Results")

st.subheader("1. Add your application materials")
left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.markdown("#### Resume")
        resume_method = st.segmented_control("Resume source", ["Upload PDF", "Paste text"], default="Upload PDF")
        resume_name = None
        if resume_method == "Upload PDF":
            uploaded_resume = st.file_uploader("Choose a PDF resume", type=["pdf"], key="resume_pdf")
            resume_text = extract_text_from_pdf(uploaded_resume) if uploaded_resume else ""
            resume_name = uploaded_resume.name if uploaded_resume else None
        else:
            resume_text = st.text_area("Resume text", height=280, placeholder="Paste the complete resume here…")
        if resume_text:
            st.markdown(f'<div class="status-ready">✓ Resume ready · {len(resume_text.split()):,} words</div>', unsafe_allow_html=True)
            with st.expander("Preview extracted text"):
                st.text(resume_text[:7000])

with right:
    with st.container(border=True):
        st.markdown("#### Job description")
        jd_method = st.segmented_control("Job-description source", ["Upload PDF", "Paste text"], default="Paste text")
        if jd_method == "Upload PDF":
            uploaded_jd = st.file_uploader("Choose a PDF job description", type=["pdf"], key="jd_pdf")
            jd_text = extract_text_from_pdf(uploaded_jd) if uploaded_jd else ""
        else:
            jd_text = st.text_area("Job-description text", height=280, placeholder="Paste the complete job description here…")
        if jd_text:
            st.markdown(f'<div class="status-ready">✓ Job description ready · {len(jd_text.split()):,} words</div>', unsafe_allow_html=True)
            with st.expander("Preview extracted text"):
                st.text(jd_text[:7000])

if resume_text and jd_text:
    ats = ats_keyword_analysis(resume_text, jd_text)
    st.subheader("2. ATS keyword snapshot")
    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Keyword coverage", f"{ats['score']}%")
        m2.metric("Matched", len(ats["matched"]))
        m3.metric("Potential gaps", len(ats["missing"]))
        st.progress(ats["score"] / 100)
        kw1, kw2 = st.columns(2)
        with kw1:
            st.markdown("**Detected in resume**")
            st.write(", ".join(ats["matched"]) or "No high-priority keyword matches detected.")
        with kw2:
            st.markdown("**Review before adding**")
            st.write(", ".join(ats["missing"]) or "No high-priority gaps detected.")
        st.caption("Only add a missing keyword when it truthfully reflects your skills or experience.")

    st.subheader("3. Choose an AI workflow")
    b1, b2, b3 = st.columns(3)
    action = None
    with b1:
        if st.button("Analyze match", type="primary", use_container_width=True):
            action = "match"
    with b2:
        if st.button("Improve resume", use_container_width=True):
            action = "enhancement"
    with b3:
        if st.button("Draft cover letter", use_container_width=True):
            action = "cover_letter"

    if action:
        labels = {"match": "Analyzing fit", "enhancement": "Building recommendations", "cover_letter": "Drafting cover letter"}
        try:
            with st.spinner(f"{labels[action]}…"):
                st.session_state.results[action] = run_workflow(action, resume_text, jd_text)
        except Exception as exc:
            logger.exception("Workflow failed")
            st.error(f"The workflow could not finish: {exc}")

    if st.session_state.results:
        st.subheader("4. Results")
        tabs = st.tabs(["Match analysis", "Resume recommendations", "Cover letter"])
        keys = ["match", "enhancement", "cover_letter"]
        empty_messages = ["Run Match Analysis to populate this tab.", "Run Improve Resume to populate this tab.", "Run Draft Cover Letter to populate this tab."]
        for tab, key, empty_message in zip(tabs, keys, empty_messages):
            with tab:
                if st.session_state.results.get(key):
                    st.markdown(st.session_state.results[key])
                else:
                    st.info(empty_message)

        report = build_report(resume_name, ats, st.session_state.results)
        st.download_button(
            "Download consolidated report",
            data=io.BytesIO(report.encode("utf-8")),
            file_name="resume_job_match_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
else:
    st.info("Add both a resume and a job description to begin.")

st.divider()
st.markdown('<p class="small-note">AI output should be reviewed before use. The app must not invent experience, skills, or credentials.</p>', unsafe_allow_html=True)
