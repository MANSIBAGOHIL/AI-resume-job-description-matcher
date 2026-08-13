# CareerFit AI — Resume–Job Description Matcher

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-7C3AED)](https://www.crewai.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-111827)](https://ollama.com/)

An AI-powered application that compares a resume with a job description using a multi-agent CrewAI workflow. It identifies alignment and keyword gaps, suggests truthful resume improvements, and generates a tailored cover letter through a locally running Ollama model.

## Technologies

- `Python`
- `CrewAI`
- `Ollama`
- `Llama 3`
- `Streamlit`
- `PyPDF2`
- `Graphviz`

## Features

- Upload PDF files or paste text for both the resume and job description
- Analyze resume–job alignment through specialized AI agents
- Display an explainable ATS keyword-coverage snapshot
- Identify matched keywords and potential keyword gaps
- Generate truthful, job-specific resume improvement suggestions
- Create a tailored cover letter using information from the resume
- Visualize the multi-agent workflow through Graphviz
- Preserve generated results while interacting with the application
- Download a consolidated Markdown report containing the available results
- Run locally with Ollama instead of requiring a hosted language-model API

## Multi-Agent Workflow

The application uses specialized agents for different parts of the analysis:

- **Resume Parsing Agent** — extracts skills, education, and work experience
- **Job Description Understanding Agent** — identifies responsibilities and required or preferred qualifications
- **Matching Agent** — evaluates the candidate's alignment with the position
- **Resume Enhancer Agent** — recommends relevant and truthful resume improvements
- **Cover Letter Agent** — produces a cover letter tailored to the role

## How the Project Was Built

1. Defined separate responsibilities for resume parsing, job-description analysis, candidate matching, resume enhancement, and cover-letter writing.
2. Created a specialized CrewAI agent for each responsibility.
3. Connected the agents through sequential CrewAI tasks that pass parsed resume and job-description context to the appropriate specialist.
4. Configured Llama 3 through Ollama so the language-model workflows run locally.
5. Built a Streamlit interface supporting both PDF uploads and pasted text.
6. Used PyPDF2 to extract text from uploaded resume and job-description files.
7. Added a deterministic keyword comparison to show matched terms and potential gaps independently of the AI-generated analysis.
8. Added persistent session results, organized output tabs, and a downloadable consolidated report.
9. Used Graphviz to present a visual overview of the multi-agent workflow.

## What I Learned

- How to divide one AI workflow among agents with specialized roles
- How CrewAI tasks pass contextual outputs between agents
- How to connect a locally running Ollama model to a multi-agent application
- How to build an interactive AI interface with Streamlit
- How to extract and process text from uploaded PDF documents
- How to combine deterministic keyword analysis with generative AI output
- How to preserve results across Streamlit reruns using session state
- How to design prompts that discourage invented skills, experience, and credentials
- How to provide downloadable results from a Streamlit application

## Project Structure

```text
MultiAgents-with-CrewAI-ResumeJDMatcher/
├── CrewaiAgents/
│   ├── __init__.py
│   ├── ResumeParsingAgent.py
│   ├── JDUnderstandingAgent.py
│   ├── MatchingAgent.py
│   ├── ResumeEnhancerAgent.py
│   └── CoverLetterAgent.py
├── ResumeJDMatchApp.py
├── requirements.txt
├── .env
└── README.md
```

## Running the Project

1. Clone the repository and open the project folder:

   ```bash
   git clone <your-repository-url>
   cd MultiAgents-with-CrewAI-ResumeJDMatcher
   ```

2. Create and activate a virtual environment:

   **Windows PowerShell**

   ```powershell
   python -m venv job_assistant
   .\job_assistant\Scripts\Activate.ps1
   ```

   **macOS/Linux**

   ```bash
   python3 -m venv job_assistant
   source job_assistant/bin/activate
   ```

3. Install the Python dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Install [Ollama](https://ollama.com/) and download the required model:

   ```bash
   ollama pull llama3:instruct
   ```

5. Make sure Ollama is running, then start the application:

   ```bash
   python -m streamlit run ResumeJDMatchApp.py
   ```

6. Open `http://localhost:8501` if the application does not open automatically.

## How to Use

1. Upload or paste a resume.
2. Upload or paste a job description.
3. Review the ATS keyword snapshot.
4. Select **Analyze Match**, **Improve Resume**, or **Draft Cover Letter**.
5. Review the generated content and download the consolidated report if needed.

## Possible Improvements

- Compare multiple resume versions against the same position
- Store previous analyses in an application-history dashboard
- Add weighted scoring for required skills, preferred skills, education, and experience
- Rewrite individual resume bullets with before-and-after comparisons

## Preview

![CareerFit AI Preview](assets/careerfit-ai-preview.png)

#### Important Note

The keyword-coverage percentage is an explainable text-overlap indicator, not an official ATS score or a guarantee of selection. All AI-generated recommendations and application materials should be reviewed before use.
