# Zenith Field-Service Assistant

A grounded RAG assistant that helps a level-1 support agent diagnose **Zenith Home Appliances** faults from the official docs. It gives step-by-step guidance with a citation, escalates safety-critical cases to a technician, and abstains when the answer isn't in the knowledge base.

Built with LangChain and Qdrant, served through Streamlit.

> Built entirely by me, except the Streamlit front end, which was AI-generated. My work is the Python backend and the whole RAG pipeline.

> Note: Zenith Home Appliances and all the data here are fictional — made up for this project, so the knowledge base is safe to share publicly.

---

## Tech stack

| Layer | Choice |
|---|---|
| Front end | Streamlit |
| Orchestration | LangChain |
| Vector store | Qdrant — in-memory (`:memory:`), rebuilt on every run |
| Dense embeddings | `BAAI/bge-base-en-v1.5` (runs locally) |
| Sparse / keyword | BM25 (fastembed) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq Cloud (temperature = 0) |
| CSV handling | pandas |
| Query logging | Google Sheets |

---

## Setup

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure

Before running, fill in a few placeholders:

| File | What to add |
|---|---|
| `rag.py` | Base path → set it to `knowledge_base/` (the folder with the KB files) |
| `rag.py` | Groq API key |
| `rag.py` | Google Sheets `credentials.json` path and sheet ID (for logging) |
| `streamlit_frontend.py` | Same Google Sheets `credentials.json` path and sheet ID |

Keep your real `credentials.json` out of the repo — `.gitignore` already blocks it.

### 3. Run

**Web app (Streamlit):**

```bash
streamlit run streamlit_frontend.py
```

**Terminal version (backend only):**

```bash
python rag.py
```

Qdrant runs in temporary in-memory mode, so the knowledge base is re-ingested and re-embedded every time the app starts.

---

## Project layout

```
rag.py                   # ingestion + retrieval + generation pipeline (terminal app)
streamlit_frontend.py    # Streamlit UI (reuses rag.py's objects)
knowledge_base/          # the six knowledge-base files
    KB_manual_split_ac.md
    KB_manual_washing_machine.md
    KB_manual_refrigerator.md
    KB_fault_codes.md
    KB_safety_bulletin_SB-2026-03.md
    KB_service_history.csv
requirements.txt
```

---

## How it works

The system has an **ingestion phase** that builds the vector index (runs fresh every session, since Qdrant is in-memory) and a **query phase** that runs per request.

### Ingestion

1. The five markdown manuals/references are loaded directly.
2. The service-history CSV is read row by row with pandas, so one appliance's ticket doesn't get blended with another's.
3. Prose is chunked with a small size and overlap (the docs are only ~2 pages each).
4. Spec and error-code tables are kept whole instead of chunked, so their row/column context isn't fragmented; the heading above each table is carried with it.
5. Chunks are embedded and loaded into the in-memory Qdrant collection.
6. Every chunk carries metadata (source document, appliance type) — that's what lets the answer layer prefer a safety bulletin over a manual when they conflict.

### Query

7. The appliance name is pulled from the query and used as a hard metadata filter, so a washer question never retrieves AC chunks (codes overlap across products — `E5` ≠ `F5`).
8. The filtered candidates go through dense + BM25 retrieval fused with RRF, then a cross-encoder reranks the top candidates.
9. The top chunks and the query go to the LLM, which returns a structured result: decision, reason, response, and citations.
10. Each query and its result are logged to a Google Sheet for review.

### Decision logic

| Outcome | When | Result |
|---|---|---|
| **Answer** | Fix is in the retrieved chunks and not safety-critical | Grounded steps + citation |
| **Escalate** | Safety-critical or technician-only fault (e.g. AC `E3` refrigerant leak, `E4` overload, gas smell/hissing) | Safety steps first, then route to a technician |
| **Abstain** | Answer not in the KB (out-of-scope product, or a fact the docs don't contain) | Says so plainly — no fabrication |

- **Grounding & citations** — every claim is tied to a retrieved source (document + section/code); answering from model memory is forbidden.
- **Conflict resolution** — the newer authoritative source wins. Example: **SB-2026-03** supersedes the manual's F4 "cool and restart" step for AquaWash units made before Jan 2025 — those must be powered off, unplugged, and repaired by a technician.
