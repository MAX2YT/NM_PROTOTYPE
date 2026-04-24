"""
app.py — Neuro-Vault Streamlit Application.

Privacy-Preserving Clinical AI for Tamil Nadu Hospitals.
Provides a full-featured clinical Q&A interface with:
  - First-run setup wizard (index build steps)
  - Hybrid RAG query pipeline
  - Dataset-aware source badges
  - Structured answer rendering (SUMMARY/EVIDENCE/SOURCES/CAUTION)
  - Tamil language support
  - Audit trail
  - Analytics dashboard
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Generator, List, Optional

import streamlit as st

# ── Page config must be first Streamlit call ──────────────────────────
st.set_page_config(
    page_title="Neuro-Vault | Clinical AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from security.audit_log import AuditLogger
from security.encryption import EncryptionManager
from tamil.translator import TamilTranslator


def svg_icon(path_d: str) -> str:
    """Return an inline SVG icon markup."""
    return (
        f'<svg class="nv-icon" viewBox="0 0 24 24" fill="none" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        f'<path d="{path_d}" stroke="currentColor" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


ICON_PATHS = {
    "brain": "M10 4a3 3 0 0 0-3 3v1a3 3 0 0 0-2 2.9V12a3 3 0 0 0 2 2.82V16a3 3 0 0 0 3 3h1m2-15a3 3 0 0 1 3 3v1a3 3 0 0 1 2 2.9V12a3 3 0 0 1-2 2.82V16a3 3 0 0 1-3 3h-1M12 4v16m-3-9h3m0 0h3",
    "summary": "M4 6h16M4 12h16M4 18h10",
    "evidence": "M9 11a3 3 0 1 1 6 0c0 1.3-.84 2.2-1.7 3.1-.7.74-1.3 1.39-1.3 2.4M12 20h.01",
    "sources": "M4 19.5A2.5 2.5 0 0 1 6.5 17H20M6.5 5H20v14H6.5A2.5 2.5 0 0 0 4 21V7.5A2.5 2.5 0 0 1 6.5 5z",
    "caution": "M12 3l9 16H3L12 3zm0 6v4m0 4h.01",
    "user": "M20 21a8 8 0 1 0-16 0M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z",
    "shield": "M12 3l7 3v6c0 4.5-3.2 8.1-7 9-3.8-.9-7-4.5-7-9V6l7-3z",
    "database": "M4 7c0-1.66 3.58-3 8-3s8 1.34 8 3-3.58 3-8 3-8-1.34-8-3zm0 5c0 1.66 3.58 3 8 3s8-1.34 8-3m-16 5c0 1.66 3.58 3 8 3s8-1.34 8-3",
    "search": "M11 19a8 8 0 1 1 0-16 8 8 0 0 1 0 16zm10 2-4.2-4.2",
    "spark": "M12 3l1.8 4.2L18 9l-4.2 1.8L12 15l-1.8-4.2L6 9l4.2-1.8L12 3z",
    "chart": "M4 20h16M7 16v-5m5 5V8m5 8v-3",
    "info": "M12 8h.01M11 12h2v6h-2zM12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z",
    "arrow": "M5 12h14m-6-6 6 6-6 6",
    "check": "M20 6 9 17l-5-5",
    "timer": "M12 8v5l3 2M12 3a9 9 0 1 0 9 9",
}


def icon(name: str) -> str:
    """Lookup SVG icon by name."""
    return svg_icon(ICON_PATHS[name])


JUDGE_DEMO_SCENARIOS = [
    (
        "Clinical Excellence",
        "How is pulmonary tuberculosis diagnosed and treated?",
        "High-quality structured response with multiple sources."
    ),
    (
        "Native Language Support",
        "நீரிழிவு நோய்க்கான சிகிச்சை முறை என்ன?",
        "Real-time Tamil translation and Tamil response."
    ),
    (
        "Abstention & Safety",
        "What is the stock price of Apollo Hospitals today?",
        "Proactively rejects non-clinical queries to ensure safety."
    ),
    (
        "Drug Interaction Warning",
        "What medications interact with Warfarin?",
        "Triggers critical 'CAUTION' alerts for medication safety."
    ),
]

# ── Session state init ────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False
if "dataset_stats" not in st.session_state:
    st.session_state.dataset_stats = {}
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"

# ── Lazy-loaded singletons ────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Neuro-Vault index…")
def load_pipeline():
    """Load and cache the full RAG pipeline (embedder, retriever, LLM)."""
    from ingest.embedder import Embedder
    from retrieval.retriever import HybridRetriever
    from retrieval.abstention import AbstentionChecker
    from llm.ollama_client import OllamaClient

    embedder = Embedder(
        model_name=Config.EMBEDDING_MODEL,
        model_dir=Config.EMBEDDING_MODEL_DIR,
        index_path=Config.FAISS_INDEX_PATH,
        metadata_path=Config.METADATA_PATH,
        batch_size=Config.EMBED_BATCH_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
    )
    embedder.load()
    retriever = HybridRetriever(embedder)
    abstention_checker = AbstentionChecker()
    llm_client = OllamaClient()
    return embedder, retriever, abstention_checker, llm_client


@st.cache_resource(show_spinner=False)
def get_audit_logger():
    enc = EncryptionManager()
    return AuditLogger(enc_mgr=enc)


@st.cache_resource(show_spinner=False)
def get_translator():
    return TamilTranslator()


# ──────────────────────────────────────────────────────────────────────
#  Custom CSS
# ──────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown(
        """
<style>
  /* ── Global ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg: #0b1220;
    --bg-elevated: #101a2b;
    --surface: #132238;
    --surface-alt: #172b45;
    --text-primary: #e6edf3;
    --text-secondary: #9db0c9;
    --border: #243a58;
    --brand: #4f8cff;
    --brand-strong: #2f6de6;
    --success: #35c47b;
    --warning: #f6b84f;
    --danger: #ea6a6a;
    --radius: 12px;
  }

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Core surfaces ── */
  .stApp {
    background: radial-gradient(circle at top right, #1a2f4f 0%, var(--bg) 55%);
    color: var(--text-primary);
  }

  .block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 1.4rem;
  }

  /* ── Tabs Fix ── */
  [data-testid="stTabs"] {
    margin-top: 0.5rem;
    background: transparent;
  }

  [data-testid="stTabs"] button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: var(--text-secondary) !important;
  }

  [data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--brand) !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1a2d 0%, #0b1424 100%);
    border-right: 1px solid var(--border);
  }

  [data-testid="stSidebar"] .stMarkdown p,
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary);
  }

  .section-panel {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: rgba(19, 34, 56, 0.75);
    padding: 0.9rem 1rem;
  }

  /* ── Chat message containers ── */
  .user-msg {
    background: linear-gradient(150deg, #173154, #142846);
    border: 1px solid #2d4f7d;
    border-radius: var(--radius);
    padding: 14px 16px;
    margin: 8px 0;
    border-left: 4px solid var(--brand);
  }

  .assistant-msg {
    background: linear-gradient(140deg, #121f34, #101a2c);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
    margin: 8px 0;
    border-left: 4px solid #6b78f5;
  }

  /* ── Answer sections ── */
  .section-summary {
    background: #10233c;
    border: 1px solid #275182;
    border-radius: 10px;
    padding: 14px;
    margin: 10px 0;
    border-left: 4px solid var(--brand);
  }

  .section-evidence {
    background: #102f27;
    border: 1px solid #216749;
    border-radius: 10px;
    padding: 14px;
    margin: 10px 0;
    border-left: 4px solid var(--success);
  }

  .section-sources {
    background: #1b1835;
    border: 1px solid #4f4f9c;
    border-radius: 10px;
    padding: 14px;
    margin: 10px 0;
    border-left: 4px solid #8288f9;
  }

  .section-caution {
    background: #352614;
    border: 1px solid #88613a;
    border-radius: 10px;
    padding: 14px;
    margin: 10px 0;
    border-left: 4px solid var(--warning);
  }

  .abstained-msg {
    background: #38181f;
    border: 1px solid #8c3f4f;
    border-radius: 10px;
    padding: 14px;
    margin: 10px 0;
    border-left: 4px solid var(--danger);
  }

  /* ── Dataset badges ── */
  .badge-mtsamples {
    background: #163b6e;
    color: #bbdcff;
    border: 1px solid #2f5f99;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 4px;
  }

  .badge-medquad {
    background: #2f2859;
    color: #d3cbff;
    border: 1px solid #5a4ca2;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 4px;
  }

  .badge-pubmed {
    background: #123425;
    color: #bdf7d7;
    border: 1px solid #2a684c;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 4px;
  }

  /* ── Dataset status indicators ── */
  .ds-active {
    color: var(--success);
    font-weight: 600;
  }

  .ds-inactive {
    color: #6b7280;
    font-style: italic;
  }

  /* ── Quick action buttons ── */
  .stButton > button {
    background: linear-gradient(140deg, #1a2d49, #183252);
    color: var(--text-primary);
    border: 1px solid #2c476c;
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    transition: all 0.2s ease;
    min-height: 42px;
  }

  .stButton > button:hover {
    background: linear-gradient(140deg, #234472, #1f4b80);
    border-color: #5c90e5;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(79, 140, 255, 0.25);
  }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: linear-gradient(145deg, #111e32, #0f192b);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
  }

  /* ── Input box ── */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
    background: var(--bg-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 8px;
  }

  /* ── Source card ── */
  .source-card {
    background: #101d31;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    margin: 6px 0;
    font-size: 0.85rem;
  }

  /* ── Header gradient text ── */
  .gradient-title {
    background: linear-gradient(135deg, #87b4ff, #8f9bff 50%, #77dba8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
  }

  .subtitle-text {
    color: var(--text-secondary);
    font-size: 0.92rem;
    margin-top: 4px;
    margin-bottom: 0.8rem;
  }

  /* ── Setup wizard steps ── */
  .wizard-step {
    background: linear-gradient(135deg, #111f33, #10253d);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin: 12px 0;
  }

  .step-done {
    border-left: 4px solid #3fb950;
  }

  .step-pending {
    border-left: 4px solid #4d5f7a;
  }

  .nv-icon {
    width: 1rem;
    height: 1rem;
    vertical-align: -0.15rem;
    margin-right: 0.35rem;
    color: currentColor;
  }

  .nv-title {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
  }

  .nv-muted {
    color: var(--text-secondary);
  }

  .hero-shell {
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.85rem 1rem 0.75rem;
    background: linear-gradient(120deg, rgba(22, 45, 74, 0.9), rgba(18, 32, 54, 0.8));
    margin-bottom: 0.7rem;
    border-left: 5px solid var(--brand);
  }

  .hero-kpis {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-top: 0.6rem;
  }

  .hero-chip {
    border: 1px solid #31557f;
    border-radius: 999px;
    padding: 0.25rem 0.55rem;
    font-size: 0.78rem;
    color: #c7dbf7;
    background: rgba(18, 41, 68, 0.8);
  }

  .trust-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.65rem 0.85rem;
    background: rgba(16, 28, 48, 0.65);
    margin-bottom: 0.5rem;
    transition: all 0.2s ease;
  }

  .trust-card:hover {
    background: rgba(22, 45, 74, 0.8);
    border-color: var(--brand);
  }

  .trust-title {
    color: #dce9ff;
    font-weight: 600;
    font-size: 0.9rem;
  }

  .trust-sub {
    color: var(--text-secondary);
    font-size: 0.8rem;
    margin-top: 0.25rem;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #161b22; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #4d9de0; }

  /* ── Mobile optimization ── */
  @media (max-width: 900px) {
    .block-container {
      padding-top: 0.8rem;
      padding-left: 0.8rem;
      padding-right: 0.8rem;
      max-width: 100%;
    }

    .gradient-title {
      font-size: 1.6rem;
      line-height: 1.25;
    }

    [data-testid="column"] {
      width: 100% !important;
      flex: 1 1 100% !important;
      min-width: 100% !important;
    }

    [data-testid="stExpander"] {
      border-radius: 10px;
    }

    .wizard-step,
    .user-msg,
    .assistant-msg {
      padding: 12px;
    }

    .hero-shell {
      padding: 0.8rem;
    }
  }
</style>
""",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────
#  Helper: render structured answer
# ──────────────────────────────────────────────────────────────────────


def render_structured_answer(answer_text: str) -> None:
    """Parse and render a structured clinical answer (SUMMARY/EVIDENCE/SOURCES/CAUTION).

    Args:
        answer_text: Raw LLM output string.
    """
    import re

    sections = {
        "SUMMARY": "",
        "EVIDENCE": "",
        "SOURCES": "",
        "CAUTION": "",
    }

    current = None
    lines = answer_text.split("\n")
    buffer: List[str] = []

    for line in lines:
        stripped = line.strip()
        matched = False
        for key in sections:
            if stripped.startswith(f"{key}:") or stripped == key:
                if current and buffer:
                    sections[current] = "\n".join(buffer).strip()
                current = key
                remainder = stripped[len(key) + 1:].strip() if ":" in stripped else ""
                buffer = [remainder] if remainder else []
                matched = True
                break
        if not matched and current:
            buffer.append(line)

    if current and buffer:
        sections[current] = "\n".join(buffer).strip()

    # Fallback: if no sections detected, display as plain text
    if not any(sections.values()):
        st.markdown(answer_text)
        return

    if sections["SUMMARY"]:
        st.markdown(
            f'<div class="section-summary">'
            f"<strong>{icon('summary')}SUMMARY</strong><br><br>{sections['SUMMARY']}</div>",
            unsafe_allow_html=True,
        )

    if sections["EVIDENCE"]:
        st.markdown(
            f'<div class="section-evidence">'
            f"<strong>{icon('evidence')}EVIDENCE</strong><br><br>{sections['EVIDENCE']}</div>",
            unsafe_allow_html=True,
        )

    if sections["SOURCES"]:
        st.markdown(
            f'<div class="section-sources">'
            f"<strong>{icon('sources')}SOURCES</strong><br><br>{sections['SOURCES']}</div>",
            unsafe_allow_html=True,
        )

    if sections["CAUTION"]:
        st.markdown(
            f'<div class="section-caution">'
            f"<strong>{icon('caution')}CAUTION</strong><br><br>{sections['CAUTION']}</div>",
            unsafe_allow_html=True,
        )


def render_source_card(chunk: dict, rank: int) -> None:
    """Render a source citation card for one retrieved chunk.

    Args:
        chunk: Retrieved chunk dict.
        rank:  1-based rank in result list.
    """
    dataset = chunk.get("dataset", "unknown")
    title = chunk.get("title", "")[:80]
    doc_type = chunk.get("doc_type", "")
    rerank = chunk.get("rerank_score", 0.0)
    journal = chunk.get("journal", "")
    year = chunk.get("year", "")
    qtype = chunk.get("qtype", "")

    # Badge
    badge_map = {
        "mtsamples": ('badge-mtsamples', 'MTSamples'),
        "medquad": ('badge-medquad', 'MedQuAD'),
        "pubmed": ('badge-pubmed', 'PubMed'),
    }
    badge_cls, badge_txt = badge_map.get(dataset, ('badge-mtsamples', dataset))

    detail_parts = []
    if doc_type:
        detail_parts.append(doc_type)
    if journal:
        detail_parts.append(journal)
    if year:
        detail_parts.append(year)
    if qtype and qtype != "general":
        detail_parts.append(qtype)
    detail = " | ".join(detail_parts)

    st.markdown(
        f'<div class="source-card">'
        f'<span class="{badge_cls}">{badge_txt}</span> '
        f'<strong>#{rank}</strong> — {title}<br>'
        f'<small style="color:#8b949e">{detail}</small> '
        f'<small style="float:right;color:#4d9de0">score: {rerank:.3f}</small>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────
#  First-run Setup Wizard
# ──────────────────────────────────────────────────────────────────────


def render_setup_wizard() -> None:
    """Render the step-by-step first-run data setup wizard."""
    st.markdown(
        f'<p class="gradient-title nv-title">{icon("brain")}Neuro-Vault</p>'
        '<p class="subtitle-text">Privacy-Preserving Clinical AI · First Run Setup</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.info(
        "Welcome to **Neuro-Vault**. No index found. "
        "Complete the four setup steps below to build your local clinical AI."
    )

    dataset_status = Config.dataset_status()

    # ── Step 1: MTSamples ────────────────────────────────────────────
    mt_done = dataset_status["mtsamples"]
    step1_cls = "wizard-step step-done" if mt_done else "wizard-step step-pending"

    with st.container():
        st.markdown(
            f'<div class="{step1_cls}">'
            f"<strong>{'DONE' if mt_done else 'PENDING'} · Step 1: MTSamples Clinical Transcriptions</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if mt_done:
            st.success(f"Found: `{Config.MTSAMPLES_PATH}`")
        else:
            st.markdown(
                "1. Visit: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions\n"
                "2. Download and extract `mtsamples.csv`\n"
                f"3. Place it at: `{Config.MTSAMPLES_PATH}`"
            )
            if st.button("Validate MTSamples file", key="btn_validate_mt",
                          use_container_width=True):
                if Config.MTSAMPLES_PATH.exists():
                    from ingest.csv_loader import validate_mtsamples
                    stats = validate_mtsamples(Config.MTSAMPLES_PATH)
                    st.success(
                        f"Validation complete. {stats['valid_rows']:,} transcription records found."
                    )
                    st.rerun()
                else:
                    st.error(f"File not found at `{Config.MTSAMPLES_PATH}`. Please check path.")

    # ── Step 2: MedQuAD ──────────────────────────────────────────────
    mq_done = dataset_status["medquad"]
    step2_cls = "wizard-step step-done" if mq_done else "wizard-step step-pending"
    st.markdown(
        f'<div class="{step2_cls}">'
        f"<strong>{'DONE' if mq_done else 'PENDING'} · Step 2: MedQuAD Medical Q&A</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if mq_done:
        mq_count = len(list(Config.MEDQUAD_DIR.rglob("*.xml")))
        st.success(f"Found {mq_count} XML files in `{Config.MEDQUAD_DIR}`")
    else:
        if st.button("Download MedQuAD (git clone)", key="btn_clone_mq",
                      use_container_width=True):
            with st.spinner("Cloning MedQuAD from GitHub (~100 MB)…"):
                import subprocess
                result = subprocess.run(
                    [
                        "git", "clone", "--depth", "1",
                        "https://github.com/abachaa/MedQuAD",
                        str(Config.MEDQUAD_DIR),
                    ],
                    capture_output=True,
                    text=True,
                )
            if result.returncode == 0:
                st.success("MedQuAD cloned successfully.")
                st.rerun()
            else:
                st.error(
                    f"Clone failed: {result.stderr}\n\n"
                    "Ensure Git is installed: https://git-scm.com/download/win"
                )

    # ── Step 3: PubMed ───────────────────────────────────────────────
    pb_done = dataset_status["pubmed"]
    step3_cls = "wizard-step step-done" if pb_done else "wizard-step step-pending"
    st.markdown(
        f'<div class="{step3_cls}">'
        f"<strong>{'DONE' if pb_done else 'PENDING'} · Step 3: PubMed Abstracts</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if pb_done:
        import json
        with open(Config.PUBMED_PATH, encoding="utf-8") as f:
            pb_count = len(json.load(f))
        st.success(f"{pb_count:,} abstracts cached at `{Config.PUBMED_PATH}`")
    else:
        st.info("Estimated time: ~3–5 minutes (20 queries × 50 abstracts each)")
        if st.button("Fetch PubMed Abstracts", key="btn_fetch_pubmed",
                      use_container_width=True):
            progress_bar = st.progress(0, text="Initialising PubMed fetch…")
            from scripts.fetch_pubmed import search_pubmed, fetch_abstracts, _parse_single_article
            import json

            queries = Config.PUBMED_QUERIES
            all_records: dict = {}
            Config.RAW_DIR.mkdir(parents=True, exist_ok=True)

            for i, query in enumerate(queries):
                progress_bar.progress(
                    i / len(queries),
                    text=f"Query {i+1}/{len(queries)}: {query[:50]}…",
                )
                pmids = search_pubmed(query)
                records = fetch_abstracts(pmids)
                for r in records:
                    if r["pmid"] not in all_records:
                        all_records[r["pmid"]] = r

            progress_bar.progress(1.0, text=f"Done! {len(all_records)} abstracts fetched.")
            with open(Config.PUBMED_PATH, "w", encoding="utf-8") as fh:
                json.dump(list(all_records.values()), fh, indent=2, ensure_ascii=False)
            st.success(f"{len(all_records):,} unique PubMed abstracts saved.")
            st.rerun()

    # ── Step 4: Build Index ──────────────────────────────────────────
    any_data = any(dataset_status.values())
    step4_cls = "wizard-step step-done" if Config.index_exists() else "wizard-step step-pending"
    st.markdown(
        f'<div class="{step4_cls}">'
        f"<strong>{'DONE' if Config.index_exists() else 'PENDING'} · Step 4: Build Vector Index</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not any_data:
        st.warning("Complete at least one dataset step above before building the index.")
    elif not Config.index_exists():
        if st.button("Build Vector Index", key="btn_build_index", use_container_width=True, type="primary"):
            _run_index_build_with_progress()
    else:
        st.success("Index ready. Refresh the page to start querying.")


def _run_index_build_with_progress() -> None:
    """Run ingestion + FAISS index build with live progress updates."""
    from ingest.csv_loader import load_mtsamples
    from ingest.xml_loader import load_medquad
    from ingest.pubmed_loader import load_pubmed
    from ingest.clinical_chunker import ClinicalAwareChunker
    from ingest.embedder import Embedder

    status = st.empty()
    progress = st.progress(0, text="Starting ingestion…")

    all_chunks: List[dict] = []
    step = 0
    total_steps = 5

    def advance(msg: str):
        nonlocal step
        step += 1
        progress.progress(step / total_steps, text=msg)
        status.markdown(f"**In progress:** {msg}")

    # MTSamples
    if Config.MTSAMPLES_PATH.exists():
        advance("Loading MTSamples…")
        docs = load_mtsamples(Config.MTSAMPLES_PATH)
        chunks = ClinicalAwareChunker(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP).chunk_documents(docs)
        all_chunks.extend(chunks)
        status.markdown(f"MTSamples: {len(docs):,} docs | {len(chunks):,} chunks")

    # MedQuAD
    if Config.MEDQUAD_DIR.exists():
        advance("Loading MedQuAD…")
        docs = load_medquad(
            Config.MEDQUAD_DIR, split="train",
            test_output_path=Config.MEDQUAD_TEST_PATH,
        )
        chunks = ClinicalAwareChunker(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP).chunk_documents(docs)
        all_chunks.extend(chunks)
        status.markdown(f"MedQuAD: {len(docs):,} docs | {len(chunks):,} chunks")

    # PubMed
    if Config.PUBMED_PATH.exists():
        advance("Loading PubMed…")
        docs = load_pubmed(Config.PUBMED_PATH)
        chunks = ClinicalAwareChunker(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP).chunk_documents(docs)
        all_chunks.extend(chunks)
        status.markdown(f"PubMed: {len(docs):,} docs | {len(chunks):,} chunks")

    if not all_chunks:
        st.error("No chunks to index! Ensure at least one dataset is available.")
        return

    # Embed + Build
    advance(f"Embedding {len(all_chunks):,} chunks (this takes several minutes)…")
    t0 = time.time()
    embedder = Embedder(
        model_name=Config.EMBEDDING_MODEL,
        model_dir=Config.EMBEDDING_MODEL_DIR,
        index_path=Config.FAISS_INDEX_PATH,
        metadata_path=Config.METADATA_PATH,
        batch_size=Config.EMBED_BATCH_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
    )
    embedder.build_index(all_chunks, show_progress=True)
    embedder.save()
    elapsed = time.time() - t0

    advance("Done!")
    progress.progress(1.0, text="Index built!")
    st.success(
        f"Index ready. **{len(all_chunks):,}** vectors indexed in **{elapsed:.0f}s**.\n\n"
        "Refresh the page to start querying."
    )


# ──────────────────────────────────────────────────────────────────────
#  Sidebar
# ──────────────────────────────────────────────────────────────────────


def render_sidebar(embedder=None) -> dict:
    """Render the sidebar with data source info, settings, and filters.

    Args:
        embedder: Loaded Embedder instance (for index stats).

    Returns:
        dict of UI settings: ``dataset_filter``, ``final_top_k``.
    """
    with st.sidebar:
        # Logo / title
        st.markdown(
            f"## {icon('brain')}Neuro-Vault\n"
            '<p style="color:#8b949e;font-size:0.8rem;">'
            "Privacy-Preserving Clinical AI<br>"
            "Tamil Nadu Hospitals Edition"
            "</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Data Sources panel ─────────────────────────────────────────
        st.markdown(f"### {icon('database')}Data Sources", unsafe_allow_html=True)
        dataset_status = Config.dataset_status()
        stats = embedder.stats() if embedder else {}
        ds_counts = stats.get("dataset_counts", {})

        def _ds_row(name: str, icon_name: str, ds_key: str) -> None:
            active = dataset_status.get(ds_key, False)
            count = ds_counts.get(ds_key, 0)
            icon_html = icon(icon_name)
            if active and count > 0:
                st.markdown(
                    f'<span class="ds-active">{icon_html}{name}</span>'
                    f'<small style="color:#3fb950;float:right">{count:,} chunks</small>',
                    unsafe_allow_html=True,
                )
            elif active:
                st.markdown(
                    f'<span class="ds-active">{icon_html}{name}</span>'
                    f'<small style="color:#d29922;float:right">Not indexed yet</small>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="ds-inactive">{icon_html}{name}</span>'
                    f'<small style="color:#6b7280;float:right">Not found</small>',
                    unsafe_allow_html=True,
                )

        _ds_row("MTSamples", "database", "mtsamples")
        _ds_row("MedQuAD", "info", "medquad")
        _ds_row("PubMed", "evidence", "pubmed")

        if any(ds_counts.values()):
            total = sum(ds_counts.values())
            st.caption(f"Total indexed: **{total:,}** vectors")

        st.divider()

        # ── Source filter ──────────────────────────────────────────────
        st.markdown(f"### {icon('search')}Search Settings", unsafe_allow_html=True)
        source_options = ["MTSamples", "MedQuAD", "PubMed"]
        active_sources = [s for s in source_options
                          if dataset_status.get(s.lower().replace("-", ""), False)]
        selected_sources = st.multiselect(
            "Search in:",
            options=source_options,
            default=active_sources if active_sources else source_options,
        )

        final_top_k = st.slider(
            "Results to retrieve", min_value=3, max_value=10,
            value=Config.FINAL_TOP_K, step=1,
        )

        demo_mode = st.toggle(
            "Judge demo mode",
            value=st.session_state.get("judge_demo_mode", False),
            help="Highlights a guided flow for live hackathon judging."
        )
        st.session_state.judge_demo_mode = demo_mode

        st.divider()

        # ── Ollama status ──────────────────────────────────────────────
        st.markdown(f"### {icon('spark')}LLM Status", unsafe_allow_html=True)
        from llm.ollama_client import OllamaClient
        llm = OllamaClient()
        if llm.is_available():
            st.success("Ollama is running")
            if llm.model_exists():
                st.success(f"Model: `{Config.OLLAMA_MODEL}`")
            else:
                st.warning(f"Model not pulled:\n```\nollama pull {Config.OLLAMA_MODEL}\n```")
        else:
            st.error(
                "Ollama is not running\n\n"
                "Start it with:\n```\nollama serve\n```"
            )

        st.divider()

        # ── Session info ──────────────────────────────────────────────
        st.markdown(
            f'<small style="color:#8b949e">'
            f"Session: `{st.session_state.session_id}`<br>"
            f"DPDP Act 2023 Compliant · 100% Offline"
            f"</small>",
            unsafe_allow_html=True,
        )

    return {
        "selected_sources": selected_sources,
        "final_top_k": final_top_k,
        "demo_mode": demo_mode,
    }


# ──────────────────────────────────────────────────────────────────────
#  Demo queries
# ──────────────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    ("Type 2 Diabetes", "What are the treatment guidelines for Type 2 Diabetes Mellitus?"),
    ("Tuberculosis", "How is pulmonary tuberculosis diagnosed and treated?"),
    ("Dengue", "What are the signs and symptoms of dengue fever?"),
    ("Tamil Query", "நீரிழிவு நோய்க்கான சிகிச்சை முறை என்ன?"),
    ("Warfarin", "What medications interact with Warfarin?"),
    ("Abstention Test", "What is the stock price of Apollo Hospitals today?"),
]


# ──────────────────────────────────────────────────────────────────────
#  Main chat interface
# ──────────────────────────────────────────────────────────────────────


def render_chat_interface(retriever, abstention_checker, llm_client, settings: dict) -> None:
    """Render the main clinical Q&A chat interface.

    Args:
        retriever:           HybridRetriever instance.
        abstention_checker:  AbstentionChecker instance.
        llm_client:          OllamaClient instance.
        settings:            dict from render_sidebar.
    """
    translator = get_translator()
    audit = get_audit_logger()
    demo_mode = settings.get("demo_mode", False)

    # ── Header ────────────────────────────────────────────────────────
    st.markdown(
        f'<p class="gradient-title nv-title">{icon("brain")}Neuro-Vault</p>'
        '<p class="subtitle-text">Clinical Decision Support · Privacy-Preserving · Offline-First</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-shell">'
        '<div style="font-size: 1.15rem; color: #fff;"><strong>Clinical-grade offline assistant for hospitals.</strong></div>'
        '<div class="nv-muted" style="margin-top:0.25rem; font-size: 0.9rem;">'
        'A secure, private alternative to public LLMs, designed for bedside decision support.'
        '</div>'
        '<div class="hero-kpis">'
        f'<span class="hero-chip">{icon("shield")} DPDP-ready</span>'
        f'<span class="hero-chip">{icon("database")} RAG Pipeline</span>'
        f'<span class="hero-chip">{icon("timer")} Low Latency</span>'
        f'<span class="hero-chip" style="background:rgba(53, 196, 123, 0.2); border-color:var(--success); color:#bdf7d7;">{icon("evidence")} 100% Offline</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    trust_col1, trust_col2, trust_col3 = st.columns(3)
    with trust_col1:
        st.markdown(
            '<div class="trust-card">'
            f'<div class="trust-title">{icon("shield")}Data never leaves network</div>'
            '<div class="trust-sub">Inference and logs remain on local infrastructure.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with trust_col2:
        st.markdown(
            '<div class="trust-card">'
            f'<div class="trust-title">{icon("sources")}Evidence-first answers</div>'
            '<div class="trust-sub">Every answer is tied to retrieved clinical sources.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with trust_col3:
        st.markdown(
            '<div class="trust-card">'
            f'<div class="trust-title">{icon("caution")}Built-in abstention</div>'
            '<div class="trust-sub">Out-of-scope queries are rejected safely.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if demo_mode:
        st.markdown(
            f'<div style="background: rgba(124, 58, 237, 0.1); border: 1px solid rgba(124, 58, 237, 0.3); border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">'
            f'<h4 style="margin-top:0; color: #c084fc;">{icon("spark")} Judge Demo Scenarios</h4>'
            f'<p style="font-size: 0.85rem; color: #d8b4fe; margin-bottom: 1rem;">Run these guided showcases to demonstrate core platform value in 60 seconds.</p>'
            '</div>',
            unsafe_allow_html=True
        )
        demo_cols = st.columns(4)
        for i, (title, demo_query, demo_purpose) in enumerate(JUDGE_DEMO_SCENARIOS):
            with demo_cols[i]:
                st.markdown(f"**{title}**")
                st.caption(demo_purpose)
                if st.button("Run Showcase", key=f"judge_demo_{i}", use_container_width=True, type="primary"):
                    st.session_state.pending_query = demo_query

    # ── Demo query buttons ────────────────────────────────────────────
    st.markdown(f"#### {icon('spark')}Quick Queries", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (label, query_text) in enumerate(DEMO_QUERIES):
        with cols[i % 3]:
            if st.button(label, key=f"demo_{i}", use_container_width=True):
                st.session_state.pending_query = query_text

    st.divider()

    # ── Chat history display ──────────────────────────────────────────
    for entry in st.session_state.chat_history:
        st.markdown(
            f'<div class="user-msg">{icon("user")} <strong>You</strong><br>{entry["query"]}</div>',
            unsafe_allow_html=True,
        )

        if entry.get("abstained"):
            st.markdown(
                f'<div class="abstained-msg">{icon("caution")} <strong>Abstained</strong><br>'
                f'{entry["reason"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.container():
                st.markdown(
                    f'<div class="assistant-msg">{icon("brain")} <strong>Neuro-Vault</strong></div>',
                    unsafe_allow_html=True,
                )
                render_structured_answer(entry.get("answer", ""))

                if entry.get("chunks"):
                    with st.expander(f"Sources ({len(entry['chunks'])} retrieved)", expanded=False):
                        for rank, chunk in enumerate(entry["chunks"], 1):
                            render_source_card(chunk, rank)
                            with st.expander(f"Context extract #{rank}", expanded=False):
                                st.markdown(
                                    f'<small style="color:#8b949e;font-family:monospace">'
                                    f'{chunk.get("text", "")[:500]}…</small>',
                                    unsafe_allow_html=True,
                                )

    # ── Query input ───────────────────────────────────────────────────
    st.divider()

    pending = st.session_state.pop("pending_query", None)
    query = st.chat_input(
        "Ask a clinical question (English or Tamil)…",
        key="chat_input",
    )
    if pending:
        query = pending

    if not query:
        return

    # ── Process query ─────────────────────────────────────────────────
    t0 = time.time()

    # Tamil detection + translation
    is_tamil = translator.is_tamil(query)
    search_query = translator.tamil_to_english(query) if is_tamil else query

    # Dataset filter from sidebar
    ds_filter_map = {
        "MTSamples": "mtsamples",
        "MedQuAD": "medquad",
        "PubMed": "pubmed",
    }
    selected = settings.get("selected_sources", [])
    # If all three selected, no filter; otherwise restrict to first selected
    dataset_filter = None
    if len(selected) == 1:
        dataset_filter = ds_filter_map.get(selected[0])

    final_top_k = settings.get("final_top_k", Config.FINAL_TOP_K)

    # Retrieve
    with st.spinner("Retrieving relevant clinical context..."):
        try:
            retriever.final_top_k = final_top_k
            chunks = retriever.retrieve(search_query, dataset_filter=dataset_filter)
        except Exception as exc:
            st.error(f"Retrieval error: {exc}")
            return

    # Abstention check
    abstain, reason = abstention_checker.should_abstain(search_query, chunks)

    if abstain:
        latency_ms = int((time.time() - t0) * 1000)
        audit.log_query(
            session_id=st.session_state.session_id,
            query=query,
            abstained=True,
            abstain_reason=reason,
            dataset_sources=[c.get("dataset", "") for c in chunks],
            num_chunks=len(chunks),
            response_len=0,
            latency_ms=latency_ms,
        )
        st.session_state.chat_history.append(
            {"query": query, "abstained": True, "reason": reason}
        )
        st.rerun()
        return

    # Generate LLM answer
    from llm.prompt_templates import build_prompt
    system_prompt, user_prompt = build_prompt(search_query, chunks, is_tamil=is_tamil)

    full_answer_parts: List[str] = []

    st.markdown(
        f'<div class="user-msg">{icon("user")} <strong>You</strong><br>{query}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="assistant-msg">{icon("brain")} <strong>Neuro-Vault</strong></div>',
        unsafe_allow_html=True,
    )

    answer_placeholder = st.empty()
    stream_buffer: List[str] = []

    with st.spinner("Generating clinical answer..."):
        try:
            for token in llm_client.generate_stream(user_prompt, system=system_prompt):
                stream_buffer.append(token)
                current_text = "".join(stream_buffer)
                answer_placeholder.markdown(current_text + "▌")
            full_answer = "".join(stream_buffer)
            answer_placeholder.empty()
            render_structured_answer(full_answer)
        except (ConnectionError, RuntimeError) as exc:
            st.error(f"LLM error: {exc}")
            full_answer = f"LLM_ERROR: {exc}"

    # Sources
    with st.expander(f"Sources ({len(chunks)} retrieved)", expanded=True):
        for rank, chunk in enumerate(chunks, 1):
            render_source_card(chunk, rank)

    # Audit
    latency_ms = int((time.time() - t0) * 1000)
    dataset_sources = list(set(c.get("dataset", "") for c in chunks))
    audit.log_query(
        session_id=st.session_state.session_id,
        query=query,
        abstained=False,
        dataset_sources=dataset_sources,
        num_chunks=len(chunks),
        response_len=len(full_answer),
        latency_ms=latency_ms,
    )

    # Append to history
    st.session_state.chat_history.append(
        {
            "query": query,
            "abstained": False,
            "answer": full_answer,
            "chunks": chunks,
        }
    )


# ──────────────────────────────────────────────────────────────────────
#  Analytics tab
# ──────────────────────────────────────────────────────────────────────


def render_analytics():
    """Render the audit trail analytics dashboard."""
    st.markdown(f"## {icon('chart')}Analytics & Audit Trail", unsafe_allow_html=True)

    audit = get_audit_logger()
    stats = audit.summary_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries", stats["total_queries"])
    c2.metric("Abstentions", stats["total_abstentions"])
    c3.metric("Abstention Rate", f"{stats['abstention_rate']:.1%}")
    c4.metric("Avg Latency", f"{stats['avg_latency_ms']:.0f} ms")

    st.divider()

    # Dataset usage
    if stats["dataset_usage"]:
        st.markdown("### Dataset Usage")
        for ds, count in stats["dataset_usage"].items():
            st.metric(ds, count)

    st.divider()

    # Recent logs
    st.markdown("### Recent Query Log")
    logs = audit.recent_logs(limit=20)
    if not logs:
        st.info("No queries recorded yet.")
    else:
        for log in logs:
            with st.expander(
                f"[{log.get('timestamp', '')[:19]}] "
                f"{'ABSTAINED' if log.get('abstained') else 'ANSWERED'} "
                f"{log.get('query', '')[:60]}…",
                expanded=False,
            ):
                st.markdown(f"**Query:** {log.get('query', '')}")
                st.markdown(f"**Abstained:** {bool(log.get('abstained'))}")
                if log.get("abstain_reason"):
                    st.markdown(f"**Reason:** {log.get('abstain_reason')}")
                st.markdown(f"**Datasets:** {', '.join(log.get('dataset_sources', []))}")
                st.markdown(f"**Latency:** {log.get('latency_ms', 0)} ms")


# ──────────────────────────────────────────────────────────────────────
#  About tab
# ──────────────────────────────────────────────────────────────────────


def render_about():
    """Render the About / DPDP compliance information page."""
    st.markdown(f"## {icon('info')}About Neuro-Vault", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### Architecture
```
User Query (Tamil/English)
        ↓
  TamilTranslator
        ↓
  BioClinicalBERT
  (Dense Embedding)
        ↓
┌─────────────────────────┐
│   Hybrid Retrieval      │
│  FAISS + BM25 → RRF    │
│  Cross-Encoder Rerank  │
└─────────────────────────┘
        ↓
 AbstentionChecker
        ↓
  Ollama (Llama 3)
        ↓
 Structured Answer
  SUMMARY / EVIDENCE
  SOURCES / CAUTION
```
""")

    with col2:
        st.markdown("""
### Data Sources

| Dataset | Source | Records |
|---------|--------|---------|
| MTSamples | Kaggle | ~5,000 transcriptions |
| MedQuAD | GitHub (NIH/CDC) | ~47,000 QA pairs |
| PubMed | NCBI E-utilities | ~1,000 abstracts |

### DPDP Act 2023 Compliance

- **MTSamples**: De-identified, public domain
- **MedQuAD**: NIH/CDC public content, no PII
- **PubMed**: Published abstracts, no PII
- **Processing**: 100% local after initial setup
- **Queries**: AES-256 encrypted in audit log
- **Zero telemetry**: No data leaves the hospital network
""")

    st.divider()
    st.markdown("""
### Quick Setup Reference

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the LLM
ollama pull llama3:8b-instruct-q4_K_M

# 3. Get datasets (see setup wizard in app)
#    OR run manually:
python scripts/download_medquad.py
python scripts/fetch_pubmed.py
python scripts/setup_datasets.py

# 4. Launch
streamlit run app.py
```
""")


# ──────────────────────────────────────────────────────────────────────
#  Main entry point
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Main Streamlit application entry point."""
    inject_css()
    Config.ensure_dirs()

    # Determine if index exists
    index_ready = Config.index_exists()

    if not index_ready:
        render_setup_wizard()
        return

    # Load pipeline (cached)
    try:
        embedder, retriever, abstention_checker, llm_client = load_pipeline()
    except FileNotFoundError as exc:
        st.error(f"Failed to load index: {exc}")
        if st.button("Re-run Setup Wizard"):
            st.session_state.clear()
            st.rerun()
        return
    except Exception as exc:
        st.error(f"Unexpected error loading pipeline: {exc}")
        st.exception(exc)
        return

    # Sidebar
    settings = render_sidebar(embedder=embedder)

    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["Clinical Q&A", "Analytics", "About"])

    with tab1:
        render_chat_interface(retriever, abstention_checker, llm_client, settings)

    with tab2:
        render_analytics()

    with tab3:
        render_about()


if __name__ == "__main__":
    main()
