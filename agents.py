import os
import fitz
from typing import List, Dict, Optional, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS

os.environ["LANGSMITH_API_KEY"]    = "YOUR-API-HERE"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "cv-assistant"
os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"

llm = ChatOllama(model="llama3.1:8b", temperature=0.2)

def read_pdf(file_bytes: bytes) -> str:
    text = ""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

class GraphState(TypedDict, total=False):
    cv_text:          str
    job_title:        str
    cv_analysis:      Optional[Dict]
    job_requirements: Optional[Dict]
    gaps:             Optional[Dict]
    improvements:     Optional[str]
    critique:         Optional[Dict]
    iteration:        int
    max_iteration:    int

ANALYZER_SYSTEM = """You are a CV Analyzer Agent.
Read the CV text carefully and extract:
- Technical and soft skills
- Work experience (company, role, duration)
- Education background
- Projects (title, description, technologies used)
- A brief professional summary
- Candidate's region

Rules:
- Extract only what is written in the CV, do not add anything
- Be specific and detailed
- Return valid JSON matching the schema
"""

JOB_SEARCH_SYSTEM = """You are a Job Search Agent.
Read the job_title and summarize the key requirements for this role.

Rules:
- Extract requirements from the search results only
- Be specific: list required skills, experience, and qualifications
- Do not add anything not mentioned in the search results
- Return valid JSON matching the schema
"""

GAP_SYSTEM = """You are a Gap Analysis Agent.
Compare the candidate CV analysis with the job requirements.

Rules:
- Treat projects as practical experience, not just academic work
- List skills and experience the candidate is missing
- List skills and experience the candidate already has
- Be specific and honest
- Return valid JSON matching the schema
"""

IMPROVER_SYSTEM = """You are a CV Improver Agent.
Based on the CV analysis and the identified gaps, write improvement suggestions.

Rules:
- Start with MISSING SKILLS section: list exactly what skills the candidate needs to add
- Then PRESENTATION section: how to present existing experience better
- Be specific: mention exact skills to add or improve
- Suggest how to quantify achievements with real numbers
- Write in clear professional English
- Do not fabricate experience the candidate does not have
"""

VALIDATOR_SYSTEM = """You are a Validator Agent.
Review the improvement suggestions and score them.

Scoring criteria:
- Score 80-100: suggestions are specific, actionable, and address the gaps
- Score 60-79: suggestions are good but missing some details
- Score 40-59: suggestions are vague or incomplete
- Score 0-39: suggestions do not address the gaps

Rules:
- Be fair and realistic
- If missing skills are mentioned AND presentation tips are given = minimum score 70
- Return valid JSON matching the schema
"""

class CVAnalysis(BaseModel):
    skills:     List[str] = Field(..., description="Technical and soft skills")
    experience: List[str] = Field(..., description="Work experience")
    education:  List[str] = Field(..., description="Education background")
    projects:   List[str] = Field(..., description="List of projects with title, description, and technologies used")
    summary:    str       = Field(..., description="Brief professional summary")
    region:     str       = Field(..., description="Candidate's location or region extracted from CV")

class JobRequirements(BaseModel):
    required_skills:    List[str] = Field(..., description="Must-have skills")
    preferred_skills:   List[str] = Field(..., description="Nice-to-have skills")
    experience_years:   str       = Field(..., description="Years of experience required")
    education_required: str       = Field(..., description="Education requirement")

class GapAnalysis(BaseModel):
    existing_skills:    List[str] = Field(..., description="Skills candidate already has")
    missing_skills:     List[str] = Field(..., description="Skills candidate is missing")
    missing_experience: List[str] = Field(..., description="Experience gaps")

class Validation(BaseModel):
    issues:           List[str] = Field(..., description="Problems with the improvements")
    score:            int       = Field(..., ge=0, le=100, description="Quality score")
    fix_instructions: List[str] = Field(..., description="How to improve")

def analyzer_node(state: GraphState) -> GraphState:
    structured_analyzer = llm.with_structured_output(CVAnalysis)
    result = structured_analyzer.invoke([
        SystemMessage(content=ANALYZER_SYSTEM),
        HumanMessage(content=state['cv_text'])
    ])
    state['cv_analysis'] = result.model_dump()
    print(f'Skills found: {len(result.skills)}')
    print(f'Projects: {len(result.projects)}')
    return state

def job_search_node(state: GraphState) -> GraphState:
    region = state['cv_analysis'].get('region', 'Saudi Arabia')
    with DDGS() as ddgs:
        raw = list(ddgs.text(
            state['job_title'] + ' job requirements ' + region + ' 2026',
            max_results=3
        ))
    context = '\n\n'.join([r.get('body', '') for r in raw])
    structured_job = llm.with_structured_output(JobRequirements)
    result = structured_job.invoke([
        SystemMessage(content=JOB_SEARCH_SYSTEM),
        HumanMessage(content=f'Job Title: {state["job_title"]}\nRegion: {region}\nSearch Results:\n{context}')
    ])
    state['job_requirements'] = result.model_dump()
    print(f'Region: {region}')
    print(f'Required skills: {len(result.required_skills)}')
    return state

def gap_node(state: GraphState) -> GraphState:
    structured_gap = llm.with_structured_output(GapAnalysis)
    result = structured_gap.invoke([
        SystemMessage(content=GAP_SYSTEM),
        HumanMessage(content=f'CV Analysis:\n{state["cv_analysis"]}\n\nJob Requirements:\n{state["job_requirements"]}')
    ])
    state['gaps'] = result.model_dump()
    print(f'Missing skills: {len(result.missing_skills)}')
    print(f'Existing skills: {len(result.existing_skills)}')
    return state

def improver_node(state: GraphState) -> GraphState:
    resp = llm.invoke([
        SystemMessage(content=IMPROVER_SYSTEM),
        HumanMessage(content=f'CV Analysis:\n{state["cv_analysis"]}\n\nJob Requirements:\n{state["job_requirements"]}\n\nGaps:\n{state["gaps"]}')
    ]).content
    state['improvements'] = resp
    return state

def validator_node(state: GraphState) -> GraphState:
    improvements = state.get('improvements', None)
    if not improvements:
        state['critique'] = {'score': 0, 'issues': [], 'fix_instructions': []}
        state['iteration'] += 1
        return state
    structured_validator = llm.with_structured_output(Validation)
    result = structured_validator.invoke([
        SystemMessage(content=VALIDATOR_SYSTEM),
        HumanMessage(content=f'Improvements:\n{improvements}\n\nCV Analysis:\n{state["cv_analysis"]}\n\nJob Requirements:\n{state["job_requirements"]}\n\nGaps:\n{state["gaps"]}')
    ])
    state['critique']  = result.model_dump()
    state['iteration'] += 1
    print(f'Validation score: {result.score} | iteration: {state["iteration"]}')
    return state

def should_revise(state: GraphState) -> Literal['revise', 'finalize']:
    score = state['critique']['score']
    if state['iteration'] >= state['max_iteration']:
        return 'finalize'
    if score < 80:
        return 'revise'
    return 'finalize'

workflow = StateGraph(GraphState)

workflow.add_node('analyzer',   analyzer_node)
workflow.add_node('job_search', job_search_node)
workflow.add_node('gap',        gap_node)
workflow.add_node('improver',   improver_node)
workflow.add_node('validator',  validator_node)

workflow.set_entry_point('analyzer')
workflow.add_edge('analyzer',   'job_search')
workflow.add_edge('job_search', 'gap')
workflow.add_edge('gap',        'improver')
workflow.add_edge('improver',   'validator')
workflow.add_conditional_edges(
    'validator',
    should_revise,
    {
        'revise':   'improver',
        'finalize': END,
    }
)

app = workflow.compile()

def run_pipeline(cv_bytes: bytes, job_title: str) -> str:
    cv_text = read_pdf(cv_bytes)
    initial_state: GraphState = {
        'cv_text':          cv_text,
        'job_title':        job_title,
        'cv_analysis':      None,
        'job_requirements': None,
        'gaps':             None,
        'improvements':     None,
        'critique':         None,
        'iteration':        0,
        'max_iteration':    2,
    }
    result = app.invoke(initial_state)
    return result['improvements']
