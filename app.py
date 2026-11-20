import os
import json
import re
import hashlib
import requests
import textwrap
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from PIL import Image
import traceback

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from ddgs import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from econml.dml import LinearDML, CausalForestDML
from econml.score import RScorer

# ==========================================
# SECTION 1: CONFIGURATION & SETUP
# ==========================================
API_KEY_FILE = ".api_key"
CACHE_DIR = "./cache/"
os.makedirs(CACHE_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "CognetAdvisor.png")
if not os.path.exists(LOGO_PATH):
    LOGO_PATH = None

DEFAULT_MODEL = "gpt-6-luna"
FALLBACK_MODEL = "gpt-3.5-turbo"
MODEL_NAME = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)

def load_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, "r") as f:
            return f.read().strip()
    return None

def save_api_key(key: str):
    with open(API_KEY_FILE, "w") as f:
        f.write(key.strip())

def clear_api_key():
    if os.path.exists(API_KEY_FILE):
        os.remove(API_KEY_FILE)
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

def get_model_name():
    try:
        if "OPENAI_MODEL" in st.secrets:
            return st.secrets["OPENAI_MODEL"]
    except Exception:
        pass
    return MODEL_NAME

def build_llm():
    primary = get_model_name()
    try:
        is_reasoning_model = primary.startswith("gpt-6") or primary.startswith("o1") or primary.startswith("o3")
        if is_reasoning_model:
            client = ChatOpenAI(model=primary) 
        else:
            client = ChatOpenAI(model=primary, temperature=0.2)
        client.invoke("ping")
        return client, primary
    except Exception as e:
        try:
            st.warning(f"⚠ Model `{primary}` unavailable. Falling back to `{FALLBACK_MODEL}`.")
        except Exception:
            pass
        return ChatOpenAI(model=FALLBACK_MODEL, temperature=0.2), FALLBACK_MODEL

def check_password() -> bool:
    try:
        stored = st.secrets["APP_PASSWORD"] if "APP_PASSWORD" in st.secrets else None
    except Exception:
        stored = None
    if not stored:
        stored = os.environ.get("APP_PASSWORD", None)
    if not stored:
        return True

    def enter():
        if st.session_state.get("password", "") == stored:
            st.session_state["password_correct"] = True
            if "password" in st.session_state:
                del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 Enter Access Password", type="password", on_change=enter, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 Enter Access Password", type="password", on_change=enter, key="password")
        st.error("😕 Password incorrect")
        return False
    return True

# ==========================================
# SECTION 2: SOURCE DETECTION & RESOURCES
# ==========================================
def get_country_domains(country: str) -> List[str]:
    m = {
        "Pakistan": ['.pk', '.edu.pk'], "USA": ['.us', '.edu'], "UK": ['.uk', '.ac.uk'],
        "Canada": ['.ca'], "Australia": ['.au', '.edu.au'], "India": ['.in', '.ac.in'],
        "Germany": ['.de'], "France": ['.fr'], "China": ['.cn', '.edu.cn'],
        "Japan": ['.jp', '.ac.jp'], "South Korea": ['.kr', '.ac.kr'],
        "Singapore": ['.sg', '.edu.sg'], "Malaysia": ['.my', '.edu.my'],
        "UAE": ['.ae', '.ac.ae'], "Saudi Arabia": ['.sa', '.edu.sa'],
        "Egypt": ['.eg', '.edu.eg'], "South Africa": ['.za', '.ac.za'],
        "Brazil": ['.br', '.edu.br'], "Mexico": ['.mx', '.edu.mx'],
        "Italy": ['.it'], "Spain": ['.es'], "Netherlands": ['.nl'],
        "Sweden": ['.se'], "Norway": ['.no'], "Russia": ['.ru'], "Other": []
    }
    return m.get(country, [])

def clean_source_url(url: str) -> str:
    if not url: return ""
    for p in ["Scraped from ", "Scraped from: ", "Scraped from - "]:
        if url.startswith(p): return url[len(p):].strip()
    return url.strip()

def detect_course_source(source_url: str, student_country: str, student_university: str = None) -> Tuple[str, str]:
    if not source_url or source_url in ['Unknown', 'Semantic estimation', 'Fallback']:
        return "semantic", "Semantic Estimation"
    sl = source_url.lower()
    cds = get_country_domains(student_country)
    uni_match = False
    if student_university:
        for part in student_university.lower().replace(' ', '').split(','):
            if part.strip() and part.strip() in sl:
                uni_match = True
                break
    if uni_match or any(d in sl for d in cds):
        return "regional", f"Regional ({student_country})"
    if sl.startswith("http"):
        return "international", "International (Foreign URL)"
    return "semantic", "Semantic Estimation"

# ==========================================
# SECTION 3: FSLSM QUESTION BANK
# ==========================================
FSLSM_QUESTIONS = [
    ("I understand something better after I (a) try it out. (b) think it through.", "ACT_REF", "a"),
    ("I would rather be considered (a) realistic. (b) innovative.", "SEN_INT", "a"),
    ("When I think about what I did yesterday, I am most likely to get (a) a picture. (b) words.", "VIS_VER", "a"),
    ("I tend to (a) understand details of a subject but may be fuzzy about its overall structure. (b) understand the overall structure but may be fuzzy about details.", "SEQ_GLO", "a"),
    ("When I am learning something new, it helps me to (a) talk about it. (b) think about it.", "ACT_REF", "a"),
    ("If I were a teacher, I would rather teach a course (a) that deals with facts and real life situations. (b) that deals with ideas and theories.", "SEN_INT", "a"),
    ("I prefer to get new information in (a) pictures, diagrams, graphs, or maps. (b) written directions or verbal information.", "VIS_VER", "a"),
    ("Once I understand (a) all the parts, I understand the whole thing. (b) the whole thing, I see how the parts fit.", "SEQ_GLO", "a"),
    ("In a study group working on difficult material, I am more likely to (a) jump in and contribute ideas. (b) sit back and listen.", "ACT_REF", "a"),
    ("I find it easier (a) to learn facts. (b) to learn concepts.", "SEN_INT", "a"),
    ("In a book with lots of pictures and charts, I am likely to (a) look over the pictures and charts carefully. (b) focus on the written text.", "VIS_VER", "a"),
    ("When I solve math problems (a) I usually work my way to the solutions one step at a time. (b) I often just see the solutions but then have to struggle to figure out the steps to get to them.", "SEQ_GLO", "a"),
    ("In classes I have taken (a) I have usually gotten to know many of the students. (b) I have rarely gotten to know many of the students.", "ACT_REF", "a"),
    ("In reading nonfiction, I prefer (a) something that teaches me new facts or tells me how to do something. (b) something that gives me new ideas to think about.", "SEN_INT", "a"),
    ("I like teachers (a) who put a lot of diagrams on the board. (b) who spend a lot of time explaining.", "VIS_VER", "a"),
    ("When I'm analyzing a story or a novel (a) I think of the incidents and try to put them together to figure out the themes. (b) I just know what the themes are when I finish reading and then I have to go back and find the incidents that demonstrate them.", "SEQ_GLO", "a"),
    ("When I start a homework problem, I am more likely to (a) start working on the solution immediately. (b) try to fully understand the problem first.", "ACT_REF", "a"),
    ("I prefer the idea of (a) certainty. (b) theory.", "SEN_INT", "a"),
    ("I remember best (a) what I see. (b) what I hear.", "VIS_VER", "a"),
    ("It is more important to me that an instructor (a) lay out the material in clear sequential steps. (b) give me an overall picture and relate the material to other subjects.", "SEQ_GLO", "a"),
    ("I prefer to study (a) in a study group. (b) alone.", "ACT_REF", "a"),
    ("I am more likely to be considered (a) careful about the details of my work. (b) creative about how to do my work.", "SEN_INT", "a"),
    ("When I get directions to a new place, I prefer (a) a map. (b) written instructions.", "VIS_VER", "a"),
    ("I learn (a) at a fairly regular pace. If I study hard, I'll 'get it.' (b) in fits and starts. I'll be totally confused and then suddenly it all 'clicks.'", "SEQ_GLO", "a"),
    ("I would rather first (a) try things out. (b) think about how I'm going to do it.", "ACT_REF", "a"),
    ("When I am reading for enjoyment, I like writers to (a) clearly say what they mean. (b) say things in creative, interesting ways.", "SEN_INT", "a"),
    ("When I see a diagram or sketch in class, I am most likely to remember (a) the picture. (b) what the instructor said about it.", "VIS_VER", "a"),
    ("When considering a body of information, I am more likely to (a) focus on details and miss the big picture. (b) try to understand the big picture before getting into the details.", "SEQ_GLO", "a"),
    ("I more easily remember (a) something I have done. (b) something I have thought a lot about.", "ACT_REF", "a"),
    ("When I have to perform a task, I prefer to (a) master one way of doing it. (b) come up with new ways of doing it.", "SEN_INT", "a"),
    ("When someone is showing me data, I prefer (a) charts or graphs. (b) text summarizing the results.", "VIS_VER", "a"),
    ("When writing a paper, I am more likely to (a) work on the beginning and progress forward. (b) work on different parts and then order them.", "SEQ_GLO", "a"),
    ("When I have to work on a group project, I first want to (a) have 'group brainstorming'. (b) brainstorm individually and then come together.", "ACT_REF", "a"),
    ("I consider it higher praise to call someone (a) sensible. (b) imaginative.", "SEN_INT", "a"),
    ("When I meet people at a party, I am more likely to remember (a) what they looked like. (b) what they said about themselves.", "VIS_VER", "a"),
    ("When I am learning a new subject, I prefer to (a) stay focused on that subject. (b) try to make connections between that subject and related subjects.", "SEQ_GLO", "a"),
    ("I am more likely to be considered (a) outgoing. (b) reserved.", "ACT_REF", "a"),
    ("I prefer courses that emphasize (a) concrete material. (b) abstract material.", "SEN_INT", "a"),
    ("For entertainment, I would rather (a) watch television. (b) read a book.", "VIS_VER", "a"),
    ("Some teachers start their lectures with an outline. Such outlines are (a) somewhat helpful. (b) very helpful.", "SEQ_GLO", "a"),
    ("The idea of doing homework in groups with one grade (a) appeals to me. (b) does not appeal to me.", "ACT_REF", "a"),
    ("When I am doing long calculations, (a) I tend to repeat all my steps and check my work carefully. (b) I find checking my work tiresome.", "SEQ_GLO", "a"),
    ("I tend to picture places I have been (a) easily and fairly accurately. (b) with difficulty and without much detail.", "VIS_VER", "a"),
    ("When solving problems in a group, I would be more likely to (a) think of the steps in the solution process. (b) think of possible consequences or applications of the solution.", "SEQ_GLO", "a"),
]

def extract_options(q_text):
    match = re.search(r'\(a\)\s*(.+?)\s*\(b\)\s*(.+)', q_text)
    if match: return match.group(1).strip(), match.group(2).strip()
    return "a", "b"

# ==========================================
# SECTION 4: UTILITIES & RAG PIPELINE
# ==========================================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def safe_llm_call(chain, input_data):
    return chain.invoke(input_data)

def build_rag_from_pdf(pdf_path: str, embeddings) -> Optional[Chroma]:
    file_hash = hashlib.md5(open(pdf_path, "rb").read()).hexdigest()[:8]
    persist_dir = f"./chroma_{file_hash}"
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
        vs.persist()
        return vs
    except Exception:
        return None

class BackgroundRAGPreparer:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.course_features = {}

    def _scrape_course_page(self, url, timeout=15):
        from bs4 import BeautifulSoup
        try:
            r = requests.get(url, timeout=timeout, verify=False, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                for el in soup(['script', 'style', 'meta', 'link', 'noscript']): el.decompose()
                text = ' '.join(soup.get_text(separator=' ').split())
                return text if len(text) > 50 else None
        except Exception:
            pass
        return None

    def _extract_features_from_context(self, course, context):
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Rate this course on Visual, Practical, Structured (0-1). Output strict JSON: C_vis, C_prac, C_struct."),
                HumanMessage(content=f"Course: {course}\nContext: {context[:2000]}")
            ])
            res = safe_llm_call(prompt | self.llm | StrOutputParser(), {})
            match = re.search(r'\{.*\}', res, re.DOTALL)
            f = json.loads(match.group()) if match else {"C_vis": 0.5, "C_prac": 0.5, "C_struct": 0.5}
            f["C_vis"] = max(0.2, min(1.0, f.get("C_vis", 0.5)))
            f["C_prac"] = max(0.1, min(1.0, f.get("C_prac", 0.5)))
            f["C_struct"] = max(0.1, min(1.0, f.get("C_struct", 0.5)))
            f["credit_hours"] = "3+1" if "lab" in course.lower() else "3+0"
            return f
        except Exception:
            return {"C_vis": 0.5, "C_prac": 0.5, "C_struct": 0.5, "credit_hours": "3+0"}

    def run(self, country, degree, courses, university=None):
        for c in courses:
            # For testing speed, we use semantic fallback if search fails
            f = self._extract_features_from_context(c, f"Standard university syllabus for {c}")
            f["source_url"] = "Semantic estimation"
            self.course_features[c] = f

# ==========================================
# SECTION 5: FSLSM AGENT
# ==========================================
class FSLSMAgent:
    def __init__(self, llm): self.answers = []
    def compute_profile(self):
        s = {"ACT_REF": 0, "SEN_INT": 0, "VIS_VER": 0, "SEQ_GLO": 0}
        for dim, d in self.answers: s[dim] += d
        visual = max(0.0, min(1.0, (s["VIS_VER"] + 11) / 22.0))
        sensing = max(0.0, min(1.0, (s["SEN_INT"] + 11) / 22.0))
        active = max(0.0, min(1.0, (s["ACT_REF"] + 11) / 22.0))
        glob = max(0.0, min(1.0, (s["SEQ_GLO"] + 11) / 22.0))
        return {"visual": visual, "sensing": sensing, "active": active, "global": glob,
                "verbal": 1.0 - visual, "intuitive": 1.0 - sensing,
                "reflective": 1.0 - active, "sequential": 1.0 - glob}

def get_dominant_poles(profile):
    dims = {
        "Input": ("Visual", profile["visual"], "Verbal", profile["verbal"]),
        "Perception": ("Sensing", profile["sensing"], "Intuitive", profile["intuitive"]),
        "Processing": ("Active", profile["active"], "Reflective", profile["reflective"]),
        "Understanding": ("Global", profile["global"], "Sequential", profile["sequential"])
    }
    return {d: (p1, s1) if s1 >= s2 else (p2, s2) for d, (p1, s1, p2, s2) in dims.items()}

# ==========================================
# SECTION 6: GRAPH VISUALIZATION
# ==========================================
def visualize_multi_course_graph(profile, course_features, comparison_results, student_name, student_country=None, student_university=None):
    G = nx.Graph()
    G.add_node(student_name, type="student")
    dom = get_dominant_poles(profile)
    ps = {"Visual": profile["visual"], "Sensing": profile["sensing"], "Active": profile["active"], "Global": profile["global"]}
    display_scores = {"Visual": dom['Input'][1], "Sensing": dom['Perception'][1], "Active": dom['Processing'][1], "Global": dom['Understanding'][1]}
    
    for p in ["Visual", "Sensing", "Active", "Global"]:
        G.add_node(p, type="pole", score=display_scores[p], label=f"{p} ({display_scores[p]:.2f})")
        G.add_edge(student_name, p, weight=ps[p], style="solid", color="gray")
        
    sc = {"regional": "#2ECC71", "international": "#3498DB", "semantic": "#F1C40F"}
    for cname, feats in course_features.items():
        st_type, _ = detect_course_source(feats.get('source_url', 'Unknown'), student_country or "Other", student_university)
        color = sc.get(st_type, "#FF5722")
        G.add_node(cname, type="course", color=color)
        p2f = {"Visual": "C_vis", "Sensing": "C_prac", "Global": "C_struct"}
        for p, fk in p2f.items():
            if fk in feats:
                ms = profile.get(p.lower(), 0.5) * feats[fk]
                if ms > 0.01: G.add_edge(p, cname, weight=ms, style="dashed", color=color)
        a_score = profile.get("active", 0.5) * 0.5
        if a_score > 0.01: G.add_edge("Active", cname, weight=a_score, style="dashed", color=color)
            
    pos = {student_name: (0, 0)}
    poles = ["Visual", "Sensing", "Active", "Global"]
    angles = np.linspace(0, 2*np.pi, len(poles), endpoint=False)
    for i, p in enumerate(poles): pos[p] = (np.cos(angles[i]), np.sin(angles[i]))
    nc = len(course_features)
    if nc > 0:
        ca = np.linspace(0, 2*np.pi, nc, endpoint=False)
        for i, c in enumerate(course_features.keys()): pos[c] = (2.2*np.cos(ca[i]), 2.2*np.sin(ca[i]))
            
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, nodelist=[student_name], node_color="#2E7D32", node_size=3500, edgecolors="black", ax=ax)
    pole_n = [n for n in G.nodes if G.nodes[n].get("type") == "pole"]
    nx.draw_networkx_nodes(G, pos, nodelist=pole_n, node_color="#5D6D7E", node_size=2200, edgecolors="black", ax=ax)
    course_n = [n for n in G.nodes if G.nodes[n].get("type") == "course"]
    cc_list = [G.nodes[n].get("color", "#FF5722") for n in course_n]
    nx.draw_networkx_nodes(G, pos, nodelist=course_n, node_color=cc_list, node_size=2500, edgecolors="black", ax=ax)
    
    solid = [(u, v) for u, v, d in G.edges(data=True) if d.get("style") == "solid"]
    if solid: nx.draw_networkx_edges(G, pos, edgelist=solid, width=2.0, edge_color="gray", ax=ax)
    dashed = [(u, v) for u, v, d in G.edges(data=True) if d.get("style") == "dashed"]
    for u, v in dashed: nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2.0, edge_color=G[u][v].get("color", "black"), style="dashed", ax=ax)
        
    labels = {n: (student_name if n == student_name else G.nodes[n].get("label", n)) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold", ax=ax)
    
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', label='Student', markerfacecolor='#2E7D32', markersize=15),
        Line2D([0], [0], color=sc["regional"], lw=4, linestyle='--', label="Regional Course"),
        Line2D([0], [0], color=sc["international"], lw=4, linestyle='--', label="International Course"),
        Line2D([0], [0], color=sc["semantic"], lw=4, linestyle='--', label="Semantic Estimation")
    ], loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title(f"Multi-Course Cognitive Overlay: {student_name}", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    return fig

@dataclass
class CognitiveGraph:
    V: List[str]
    E: List[Dict]
    X: Dict[str, List[float]]
    a: Dict[str, float]
    reasoning_trace: List[str] = field(default_factory=list)
    best_course: str = None
    student_name: str = None

    def add_reasoning_step(self, step): self.reasoning_trace.append(step)

def generate_comparative_cognitive_graph(student_name, profile, courses, course_features, comparison_results, best_course, country=None, degree=None, semester=None, university=None, n_steps=3, llm=None):
    cg = CognitiveGraph(V=[student_name], E=[], X={student_name: [0.5]*4}, a={student_name: 1.0}, best_course=best_course, student_name=student_name)
    cg.add_reasoning_step("System 1: Expanding context from student profile")
    
    # Dummy graph for testing
    cg.add_reasoning_step(f"  Comparing {len(courses)} courses.")
    for c, d in sorted(comparison_results.items(), key=lambda x: x[1]["ite"], reverse=True):
        cg.add_reasoning_step(f"  {c}: ITE = {d['ite']:.4f}")
    
    cg.add_reasoning_step(f"Final attention: {best_course}")
    return cg

# ==========================================
# SECTION 7: CAUSAL ANALYSIS (DML)
# ==========================================
REQUIRED_NUMERIC_COLS = ['visual', 'sensing', 'active', 'global', 'prior_gpa', 'C_vis', 'C_prac', 'C_struct', 'final_grade']

def get_comparative_analysis(profile, course_features, llm, student_country="Pakistan", student_university=None):
    # Verify CSV exists
    dataset_path = os.path.join(BASE_DIR, "CogNet_Advisory_Data.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset `CogNet_Advisory_Data.csv` not found. Please ensure it is uploaded to your GitHub repo.")

    df = pd.read_csv(dataset_path, on_bad_lines='skip', low_memory=False)
    df.rename(columns={'Visual': 'visual', 'Sensing': 'sensing', 'Active': 'active', 'Global': 'global', 'GPA': 'prior_gpa', 'Course_Taken': 'course_taken', 'Final_Grade': 'final_grade'}, inplace=True)
    
    for col in REQUIRED_NUMERIC_COLS:
        if col not in df.columns:
            raise ValueError(f"Dataset is missing required column: {col}")

    for col in REQUIRED_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=REQUIRED_NUMERIC_COLS)
    
    if len(df) == 0: raise ValueError("Dataset is empty after cleaning.")

    avg_vis = df['C_vis'].mean()
    X = df[['visual', 'sensing', 'active', 'global']].values
    T = df['C_vis'].values
    Y = df['final_grade'].values
    W = df[['prior_gpa']].values

    dml = None
    try:
        # Fast fallback for Streamlit Cloud
        dml = LinearDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor(), discrete_treatment=False, cv=3)
        dml.fit(Y, T, X=X, W=W)
    except Exception as e:
        print("DML failed:", e)

    student_X = np.array([[profile['visual'], profile['sensing'], profile['active'], profile['global']]])
    cate = float(dml.effect(student_X)[0]) if dml is not None else 0.0

    results = {}
    best_course, best_ite = None, -np.inf
    
    for cname, feats in course_features.items():
        legacy_y = (0.5 + 0.3 * profile['visual'] * feats.get('C_vis', 0.5)
                    + 0.2 * profile['sensing'] * feats.get('C_prac', 0.5)
                    + 0.1 * profile['global'] * feats.get('C_struct', 0.5))
        
        delta_vis = feats.get('C_vis', 0.5) - avg_vis
        causal_ite = cate * delta_vis
        
        predicted_grade = max(0, min(1, legacy_y + causal_ite))
        src_url = feats.get('source_url', 'Unknown')
        src_type, src_label = detect_course_source(src_url, student_country, student_university)
        
        results[cname] = {
            'ite': causal_ite, 'predicted_grade': predicted_grade, 'base_grade': legacy_y, 
            'C_vis': feats.get('C_vis', 0.5), 'C_prac': feats.get('C_prac', 0.5), 'C_struct': feats.get('C_struct', 0.5),
            'source_type': src_type, 'source_label': src_label, 'matches': {}
        }
        if causal_ite > best_ite:
            best_ite, best_course = causal_ite, cname

    if best_course is None: raise ValueError("No courses were processed.")

    final_exp = f"Based on the analysis, {best_course} is recommended. The expected grade impact is {best_ite*100:+.2f}%."
    
    return best_course, results, final_exp, df, avg_vis

# ==========================================
# SECTION 8: PDF REPORT GENERATION
# ==========================================
def generate_pdf_report(profile, comparison_results, explanation, cg, simple_fig, student_name, country, degree, semester, university, courses, course_features, logo_path=None, avg_vis=0.62):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.05, 0.95, "CogNet Advisor Report", fontsize=18, fontweight='bold')
        pdf.savefig(fig, dpi=100, bbox_inches='tight'); plt.close(fig)
        
        # Save the visual graph
        pdf.savefig(simple_fig, dpi=100, bbox_inches='tight')
        # DO NOT CLOSE simple_fig HERE (We need it for the UI)
    buffer.seek(0)
    return buffer

# ==========================================
# SECTION 9: MAIN STREAMLIT APP
# ==========================================
def main():
    st.set_page_config(page_title="CogNet Advisor", layout="wide")
    if not check_password(): st.stop()

    c1, c2 = st.columns([1, 5])
    with c2:
        st.title("CogNet Advisor")
        st.markdown("*Cognitive Graph-Based Personalized Course Recommendation System using FSLSM*")

    with st.sidebar:
        st.header("API Key")
        api_key = st.text_input("OpenAI API Key", type="password")
        if not api_key: 
            st.warning("Please enter your OpenAI API key."); st.stop()
        os.environ["OPENAI_API_KEY"] = api_key

        st.header("Student Profile")
        name = st.text_input("Name", "Student")
        country = st.selectbox("Country", ["Pakistan", "USA", "UK", "Canada", "Other"])
        degree = st.selectbox("Degree", ["BS Computer Science", "BS Software Engineering", "Other"])
        semester = st.selectbox("Academic Year", list(range(2020, 2031)), index=6)
        university = st.text_input("University (optional)")

        st.header("Courses")
        course1 = st.text_input("Course 1", "Artificial Intelligence")
        courses = [course1] if course1.strip() else []
            
        if st.button("Start Assessment", type="primary"):
            st.session_state["start_assessment"] = True
            st.session_state["profile"] = None
            st.session_state["best_course"] = None
            st.session_state["analysis_started"] = False
            st.rerun()

    llm, active_model = build_llm()
    st.sidebar.caption(f"🧠 Active model: `{active_model}`")
    embeddings = OpenAIEmbeddings()

    for k, v in [("start_assessment", False), ("answers", []), ("agent", FSLSMAgent(llm)), ("profile", None), ("course_features", None), ("comparison_results", None), ("best_course", None), ("explanation", None), ("cg", None), ("analysis_started", False), ("simple_fig", None), ("pdf_buffer", None)]:
        if k not in st.session_state: st.session_state[k] = v

    if st.session_state.start_assessment and st.session_state.profile is None:
        st.subheader("FSLSM Questionnaire")
        answers = {}
        with st.form(key="fslsm_form"):
            for idx, (qt, dim, pole) in enumerate(FSLSM_QUESTIONS[:5]): # Reduced to 5 questions for fast testing!
                oa, ob = extract_options(qt)
                clean = re.sub(r'\s*\(a\)\s*.*?\s*\(b\)\s*.*', '', qt).strip()
                choice = st.radio(f"Q{idx+1}: {clean}", [f"(a) {oa}", f"(b) {ob}"], key=f"q_{idx}", index=None)
                answers[idx] = choice
            sub = st.form_submit_button("Submit All Answers")
        if sub:
            if any(answers.get(i) is None for i in range(5)):
                st.warning("Please answer all questions.")
            else:
                st.session_state.answers = []
                for idx, (_, dim, pole) in enumerate(FSLSM_QUESTIONS[:5]):
                    sel = "a" if answers[idx].startswith("(a)") else "b"
                    d = 1 if sel == pole else -1
                    st.session_state.answers.append((dim, d))
                st.session_state.agent.answers = st.session_state.answers
                st.session_state.profile = st.session_state.agent.compute_profile()
                st.success("Assessment complete!")
                if st.button("Analyze"):
                    st.session_state.analysis_started = True
                    st.rerun()
                    
    elif st.session_state.analysis_started and st.session_state.profile is not None and st.session_state.best_course is None:
        st.info("Running causal analysis and RAG extraction... Please wait.")
        with st.spinner("Extracting course features and running DML..."):
            try:
                rag = BackgroundRAGPreparer(llm, embeddings)
                rag.run(country, degree, courses, university)
                
                best_course, results, exp, df, avg_vis_val = get_comparative_analysis(
                    st.session_state.profile, rag.course_features, llm, country, university
                )
                
                st.session_state.course_features = rag.course_features
                st.session_state.comparison_results = results
                st.session_state.best_course = best_course
                st.session_state.explanation = exp
                
                st.session_state.simple_fig = visualize_multi_course_graph(
                    st.session_state.profile, rag.course_features, results, name, country, university
                )
                
                cg = generate_comparative_cognitive_graph(
                    name, st.session_state.profile, courses, rag.course_features, results, best_course, country, degree, semester, university, llm=llm
                )
                st.session_state.cg = cg
                
                st.session_state.pdf_buffer = generate_pdf_report(
                    st.session_state.profile, results, exp, cg, st.session_state.simple_fig,
                    name, country, degree, semester, university, courses, rag.course_features, 
                    LOGO_PATH, avg_vis=avg_vis_val
                )
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.analysis_started = False
                st.text(traceback.format_exc())

    elif st.session_state.profile is not None and st.session_state.best_course is not None:
        st.header("Results")
        st.success(f"**Recommended Course:** {st.session_state.best_course}")
        
        st.subheader("Cognitive Graphs")
        st.pyplot(st.session_state.simple_fig)
        
        st.subheader("Master Causal Explanation")
        st.info(st.session_state.explanation)

        st.download_button("📄 Download PDF Report", data=st.session_state.pdf_buffer, file_name=f"Report_{name}.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
