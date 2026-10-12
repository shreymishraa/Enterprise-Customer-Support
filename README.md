# 🤖 Enterprise Customer Support Automation — Multi-Agent RAG Pipeline

A production-grade **Retrieval-Augmented Generation (RAG)** system with a **multi-agent workflow** built on **LangGraph**, exposed via **FastAPI**, and shipped with a **Streamlit** chat UI and **GitHub Actions CI/CD**.

---

## 🏗️ Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│           LangGraph Orchestrator             │
│                                             │
│  [Planner] → [Query Understanding Agent]   │
│       ↓                                     │
│  [RAG Retrieval Agent] ←── FAISS Store      │
│       ↓                                     │
│  [Sentiment Analysis Agent]                 │
│       ↓                                     │
│  [Validator Agent] ──→ retry if fails       │
│       ↓                                     │
│  [Escalation Agent] (if needed)             │
│       ↓                                     │
│  [Response Generator]                       │
└─────────────────────────────────────────────┘
    │
    ▼
FastAPI REST API  ←→  Streamlit Chat UI
```

---

## 📁 Project Structure

```
capstone-rag-agents/
├── data/                        # Corpus and processed chunks
│   └── customer_support_faq.json
├── notebooks/                   # Experiments and evaluation
│   └── evaluation.ipynb
├── src/
│   ├── ingestion/               # Loading, chunking, embedding
│   │   ├── __init__.py
│   │   └── ingest.py
│   ├── retrieval/               # FAISS, hybrid search
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── tools/                   # RAG tool and agent tools
│   │   ├── __init__.py
│   │   └── rag_tool.py
│   ├── agents/                  # Agent definitions and prompts
│   │   ├── __init__.py
│   │   ├── query_understanding.py
│   │   ├── rag_retrieval.py
│   │   ├── sentiment_analysis.py
│   │   └── escalation.py
│   ├── graph/                   # LangGraph workflow and state
│   │   ├── __init__.py
│   │   ├── state.py
│   │   └── workflow.py
│   └── evaluation/              # BLEU, ROUGE, relevance scripts
│       ├── __init__.py
│       └── evaluate.py
├── api/                         # FastAPI app and schemas
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── ui/                          # Streamlit chat UI
│   └── main.py
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── tests/
│   ├── test_retriever.py
│   ├── test_tools.py
│   └── test_api.py
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo>
cd capstone-rag-agents
cp .env.example .env
# Edit .env with your API keys
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Ingest Data
```bash
python -m src.ingestion.ingest
```

### 4. Run FastAPI Backend
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Run Streamlit Frontend
```bash
streamlit run ui/main.py
```

### 6. Run with Docker Compose
```bash
docker-compose up --build
```

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Submit a support query |
| `GET` | `/health` | Health check |
| `POST` | `/feedback` | Submit feedback |
| `GET` | `/docs` | Swagger UI |

---

## 🧪 Running Tests
```bash
pytest tests/ -v --cov=src --cov=api
```

---

## 📊 Evaluation Metrics

| Metric | Baseline RAG | Agentic RAG |
|--------|-------------|-------------|
| BLEU | ~0.18 | ~0.26 |
| ROUGE-1 | ~0.42 | ~0.55 |
| ROUGE-L | ~0.38 | ~0.51 |
| Relevance | ~0.71 | ~0.84 |

---

## 🌟 Features

- ✅ **Hybrid Retrieval**: Semantic (FAISS) + Keyword (BM25)
- ✅ **Multi-Agent Workflow**: LangGraph orchestration
- ✅ **Agents as Tools**: Agents invoke each other
- ✅ **Feedback Loop**: Auto-retry with query rewriting
- ✅ **Validation**: Relevance grounding check
- ✅ **Sentiment Analysis**: Urgency detection
- ✅ **Escalation Logic**: Routes to human if needed
- ✅ **FastAPI + Streamlit**: Full-stack deployment
- ✅ **Docker + GitHub Actions**: CI/CD pipeline

---

## ⚠️ Disclaimer

This system is an **AI-assisted decision support tool**. Responses are generated from a knowledge base and should be reviewed before being used as final answers in critical situations.
