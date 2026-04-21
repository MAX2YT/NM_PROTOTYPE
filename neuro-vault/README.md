# 🧠 Neuro-Vault
### Privacy-Preserving Clinical AI for Tamil Nadu Hospitals

> **100% offline · AES-256 encrypted audit trail · DPDP Act 2023 compliant**

A local Retrieval-Augmented Generation (RAG) system that answers clinical questions from three vetted medical datasets — entirely on your hospital's own hardware, with zero data leaving the premises.

---

## 🗄️ Dataset Information

| Dataset | Source | Size | Use in Neuro-Vault |
|---------|--------|------|--------------------|
| 🏥 **MTSamples** | [Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) | ~5,000 records | Clinical transcriptions corpus (surgery, radiology, psychiatry, etc.) |
| ❓ **MedQuAD** | [GitHub](https://github.com/abachaa/MedQuAD) | ~47,000 QA pairs | Medical Q&A corpus (NIH, CDC, NCI) + RAGAS evaluation ground truth |
| 🔬 **PubMed** | [NCBI E-utilities](https://pubmed.ncbi.nlm.nih.gov/) | ~1,000 abstracts | Evidence-based clinical guidelines and research |

---

## 🚀 Quick Start

### Option A: Streamlit Setup Wizard (Recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the local LLM
ollama pull llama3:8b-instruct-q4_K_M

# 3. Launch — the wizard will guide you through dataset setup
streamlit run app.py
```

The wizard will walk you through:
- Placing `mtsamples.csv` (from Kaggle)
- Auto-cloning MedQuAD from GitHub
- Fetching PubMed abstracts (requires internet, ~3–5 min)
- Building the FAISS vector index

### Option B: Manual Setup

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Pull Ollama model
ollama pull llama3:8b-instruct-q4_K_M

# Step 3a: MTSamples (manual download required)
#   Download from: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
#   Place mtsamples.csv at: data/raw/mtsamples.csv
python scripts/download_kaggle.py

# Step 3b: MedQuAD
python scripts/download_medquad.py

# Step 3c: PubMed abstracts (~3-5 min, requires internet once)
python scripts/fetch_pubmed.py

# Step 3d: Build FAISS index (all datasets combined)
python scripts/setup_datasets.py

# Step 4: Launch
streamlit run app.py
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                  │
│  ┌────────────┐  ┌──────────────────┐  ┌─────────────────────────┐ │
│  │ MTSamples  │  │    MedQuAD XML   │  │  PubMed Abstracts JSON  │ │
│  │ (CSV ~5K)  │  │  (~47K QA pairs) │  │  (NCBI E-utilities API) │ │
│  └─────┬──────┘  └────────┬─────────┘  └───────────┬─────────────┘ │
│        └─────────────────┼────────────────────────┘               │
│                          ▼                                          │
│             ┌─────────────────────────┐                            │
│             │  ClinicalAwareChunker   │                            │
│             │  (Section-aware splits) │                            │
│             └────────────┬────────────┘                            │
│                          ▼                                          │
│             ┌─────────────────────────┐                            │
│             │   BioClinicalBERT       │                            │
│             │   (768-dim embeddings)  │                            │
│             └────────────┬────────────┘                            │
│                          ▼                                          │
│             ┌─────────────────────────┐                            │
│             │   FAISS IndexFlatL2     │  ← data/vector_store/     │
│             │   + BM25 Okapi index    │                            │
│             └─────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘

                    ┌──────────────┐
   User Query ──► │ TamilTranslator│ (Tamil ↔ English)
                    └──────┬───────┘
                           ▼
              ┌────────────────────────┐
              │   HybridRetriever      │
              │  Dense (FAISS) + BM25  │
              │  → RRF Fusion          │
              │  → Cross-Encoder Rerank│
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │   AbstentionChecker    │
              │  rerank + cosine +     │
              │  entity coverage       │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │  Ollama LLM (Llama 3)  │
              │  100% local inference  │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │  Structured Answer     │
              │  SUMMARY / EVIDENCE    │
              │  SOURCES / CAUTION     │
              └────────────────────────┘
                           │
              ┌────────────▼───────────┐
              │  AuditLogger (SQLite)  │
              │  AES-256 encrypted     │
              └────────────────────────┘
```

---

## 💬 Demo Queries

| Query | Language | Expected Behaviour |
|-------|----------|-------------------|
| `What are the treatment guidelines for Type 2 Diabetes Mellitus?` | English | Full SUMMARY + EVIDENCE from PubMed + MedQuAD |
| `How is pulmonary tuberculosis diagnosed and treated?` | English | Clinical answer with MTSamples + PubMed citations |
| `நீரிழிவு நோய்க்கான சிகிச்சை முறை என்ன?` | Tamil | Answer translated back to Tamil |
| `What medications interact with Warfarin?` | English | Drug interaction from MedQuAD + PubMed |
| `What is the stock price of Apollo Hospitals?` | English | `INSUFFICIENT_CONTEXT:` — abstention triggered |

---

## 🔒 DPDP Act 2023 Compliance

All three datasets are appropriate for clinical AI under India's Digital Personal Data Protection Act 2023:

| Dataset | PII Status | Jurisdiction | Storage |
|---------|-----------|--------------|---------|
| MTSamples | De-identified, public domain | US Kaggle dataset | Local only |
| MedQuAD | NIH/CDC public health content, zero PII | US Government sources | Local only |
| PubMed | Published research abstracts, zero PII | NCBI/NLM public data | Local only |

**Processing principles:**
- ✅ All model inference: 100% local (Ollama)
- ✅ All embeddings: local BioClinicalBERT
- ✅ Query audit log: AES-256 encrypted at rest (Fernet/PBKDF2)
- ✅ Zero telemetry: no data crosses hospital network boundary
- ✅ Abstention safety: refuses to answer when context is insufficient

---

## 📁 Project Structure

```
neuro-vault/
├── app.py                         # Streamlit UI (setup wizard + chat)
├── config.py                      # All paths, models, thresholds
├── requirements.txt
├── .env.example
├── scripts/
│   ├── setup_datasets.py          # Master setup orchestrator
│   ├── download_kaggle.py         # MTSamples validator
│   ├── download_medquad.py        # MedQuAD git clone + XML parse
│   └── fetch_pubmed.py            # PubMed NCBI E-utilities fetcher
├── ingest/
│   ├── csv_loader.py              # MTSamples → document dicts
│   ├── xml_loader.py              # MedQuAD XML → document dicts
│   ├── pubmed_loader.py           # PubMed JSON → document dicts
│   ├── pdf_loader.py              # PDF → document dicts (optional)
│   ├── clinical_chunker.py        # Dataset-aware section chunker
│   └── embedder.py                # BioClinicalBERT + FAISS
├── retrieval/
│   ├── retriever.py               # Hybrid RRF + cross-encoder
│   └── abstention.py              # 3-signal abstention gating
├── llm/
│   ├── ollama_client.py           # Streaming Ollama client
│   └── prompt_templates.py        # Clinical prompt construction
├── security/
│   ├── encryption.py              # AES-256 Fernet encryption
│   └── audit_log.py               # SQLite tamper-evident audit trail
├── tamil/
│   └── translator.py              # Tamil ↔ English with offline fallback
├── eval/
│   └── ragas_eval.py              # RAGAS evaluation on MedQuAD test set
└── tests/
    ├── test_chunker.py
    ├── test_retriever.py
    └── test_abstention.py
```

---

## 🧪 Running Tests

```bash
cd neuro-vault
pytest tests/ -v
```

---

## 🧮 Evaluation

```bash
# Run RAGAS evaluation on the MedQuAD held-out test set
python -c "from eval.ragas_eval import run_evaluation; print(run_evaluation(max_samples=50))"
```

Results saved to: `data/eval/ragas_results_YYYYMMDD_HHMMSS.json`

---

## ⚙️ Configuration

All settings in `config.py`. Key overrides via `.env`:

```bash
cp .env.example .env
# Edit .env with your values
```

Key settings:
- `OLLAMA_MODEL` — LLM model name (default: `llama3:8b-instruct-q4_K_M`)
- `EMBEDDING_MODEL` — HuggingFace model (default: `emilyalsentzer/Bio_ClinicalBERT`)
- `NEURO_VAULT_KEY` — Encryption passphrase (change for production!)

---

## 🤝 Acknowledgements

- **MTSamples**: [tboyle10 on Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
- **MedQuAD**: [Asma Ben Abacha & Dina Demner-Fushman, NIH](https://github.com/abachaa/MedQuAD)
- **PubMed**: NCBI National Library of Medicine E-utilities API
- **BioClinicalBERT**: [Emily Alsentzer et al., MIT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- **Ollama**: [ollama.ai](https://ollama.ai) for local LLM serving
