"""
Zenith Field Service — Streamlit front end (display layer).

This file does NOT modify rag.py. It runs rag.py's top half once (document
loading, chunking, embedding, index build, prompts, models, constants) and
reuses those objects. rag.py stays a standalone terminal app; this stays a
standalone web app. They share nothing except the objects this file borrows
from rag.py at boot.

The query steps below (classify -> retrieve -> rerank -> compose) are a
faithful mirror of rag.py's loop(), because loop() is terminal-only (input()/
print()) and cannot be called from a browser. Every tunable is pulled from
rag.py so config lives in one place. Logging to Google Sheets is included here.
"""

import html
import json
import pathlib
import re
import time

import streamlit as st
import gspread
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, Prefetch, FusionQuery, Fusion
)

ENGINE_PATH = pathlib.Path(__file__).parent / "rag.py"

# ── logging target (edit these two before running) ──────────────────────────
LOG_CREDENTIALS = "ADD credentials.json PATH"
LOG_SHEET_KEY = "ADD GOOGLE SHEET ID"
LOG_WORKSHEET = "Sheet1"


st.set_page_config(
    page_title="Zenith Field Service",
    page_icon="◈",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Public+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
  --ink:      #14171A;
  --panel:    #1A1E22;
  --panel-2:  #1F2429;
  --rule:     #2B3137;
  --paper:    #DCE0E2;
  --muted:    #868E96;
  --teal:     #4FA893;
  --clay:     #C08A4E;
  --stone:    #7A828A;
}

.stApp {
  background:
    radial-gradient(900px 420px at 50% -180px, rgba(79,168,147,.09), transparent 70%),
    var(--ink);
  color: var(--paper);
  font-family: 'Public Sans', system-ui, sans-serif;
}

header[data-testid="stHeader"], #MainMenu, footer { display: none; }

.block-container {
  max-width: 780px;
  padding-top: 2.4rem;
  padding-bottom: 8rem;
}

/* ── masthead ─────────────────────────────────────────────── */
.mast { margin-bottom: 2.2rem; }
.mast h1 {
  font-family: 'Archivo', sans-serif;
  font-weight: 700;
  font-size: 1.55rem;
  letter-spacing: .14em;
  text-transform: uppercase;
  margin: 0;
  color: var(--paper);
}
.mast h1 span { color: var(--teal); }
.mast p {
  font-family: 'DM Mono', monospace;
  font-size: .72rem;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--muted);
  margin: .55rem 0 0;
}
.mast .rule {
  height: 1px;
  margin-top: 1.4rem;
  background: linear-gradient(90deg, var(--teal), var(--rule) 42%, transparent);
}

/* ── turns ────────────────────────────────────────────────── */
.turn { animation: rise .45s cubic-bezier(.2,.7,.3,1) both; }
@keyframes rise { from { opacity: 0; transform: translateY(10px); } }

.turn.user { display: flex; justify-content: flex-end; margin: 2.1rem 0 1.5rem; }
.turn.user .bubble {
  max-width: 82%;
  background: var(--panel-2);
  border: 1px solid var(--rule);
  border-radius: 14px 14px 3px 14px;
  padding: .7rem 1rem;
  font-size: .96rem;
  line-height: 1.55;
  color: var(--paper);
}

.turn.bot { margin-bottom: 1.2rem; }

/* ── verdict plate (signature) ────────────────────────────── */
.plate {
  display: inline-flex;
  align-items: center;
  gap: .6rem;
  padding: .42rem .85rem .42rem .7rem;
  border: 1px solid var(--rule);
  border-left: 2px solid var(--stone);
  border-radius: 3px;
  background: var(--panel);
  margin-bottom: .95rem;
  animation: stamp .5s cubic-bezier(.16,1,.3,1) both;
}
@keyframes stamp {
  from { opacity: 0; transform: translateY(-4px) scale(.97); letter-spacing: .22em; }
}
.plate-label {
  font-family: 'Archivo', sans-serif;
  font-size: .7rem;
  font-weight: 600;
  letter-spacing: .16em;
  text-transform: uppercase;
}
.plate-meta {
  font-family: 'DM Mono', monospace;
  font-size: .68rem;
  color: var(--muted);
  padding-left: .6rem;
  border-left: 1px solid var(--rule);
}
.plate-dot { width: 6px; height: 6px; border-radius: 50%; }

.plate--answer   { border-left-color: var(--teal); }
.plate--answer   .plate-label { color: var(--teal); }
.plate--answer   .plate-dot   { background: var(--teal); }
.plate--escalate { border-left-color: var(--clay); background: rgba(192,138,78,.07); }
.plate--escalate .plate-label { color: var(--clay); }
.plate--escalate .plate-dot { background: var(--clay); animation: blink 1.6s ease-in-out infinite; }
.plate--abstain  .plate-label { color: var(--stone); }
.plate--abstain  .plate-dot { background: var(--stone); }
@keyframes blink { 50% { opacity: .25; } }

.response {
  font-size: 1rem;
  line-height: 1.72;
  color: var(--paper);
  white-space: pre-wrap;
}

/* ── citations ────────────────────────────────────────────── */
.cites { margin-top: 1.1rem; display: flex; flex-wrap: wrap; gap: .4rem; }
.cite {
  font-family: 'DM Mono', monospace;
  font-size: .7rem;
  color: var(--muted);
  border: 1px solid var(--rule);
  border-radius: 2px;
  padding: .22rem .5rem;
  background: var(--panel);
}

/* ── working indicator ────────────────────────────────────── */
.work { margin: 1.4rem 0; }
.work-label {
  font-family: 'DM Mono', monospace;
  font-size: .74rem;
  letter-spacing: .1em;
  color: var(--muted);
  text-transform: lowercase;
}
.work-label::after {
  content: '';
  animation: dots 1.2s steps(4, end) infinite;
}
@keyframes dots { 0%{content:''} 25%{content:'.'} 50%{content:'..'} 75%{content:'...'} }
.work-track { height: 1px; background: var(--rule); margin-top: .7rem; overflow: hidden; }
.work-fill  { height: 1px; background: var(--teal); transition: width .5s ease; }

/* ── source cards ─────────────────────────────────────────── */
.src {
  border-left: 1px solid var(--rule);
  padding: .1rem 0 .1rem .85rem;
  margin-bottom: 1.05rem;
  animation: rise .4s ease both;
}
.src-head {
  font-family: 'DM Mono', monospace;
  font-size: .7rem;
  color: var(--muted);
  display: flex;
  gap: .8rem;
  flex-wrap: wrap;
  margin-bottom: .35rem;
}
.src-head b { color: var(--teal); font-weight: 500; }
.src-sec {
  font-family: 'Archivo', sans-serif;
  font-size: .78rem;
  letter-spacing: .04em;
  color: var(--paper);
  margin-bottom: .3rem;
}
.src-body {
  font-family: 'DM Mono', monospace;
  font-size: .74rem;
  line-height: 1.65;
  color: #A8B0B6;
  white-space: pre-wrap;
  max-height: 190px;
  overflow-y: auto;
}

/* ── expander ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: none !important;
  background: transparent !important;
}
[data-testid="stExpander"] details {
  border: 1px solid var(--rule) !important;
  border-radius: 4px;
  background: var(--panel);
}
[data-testid="stExpander"] summary p {
  font-family: 'DM Mono', monospace !important;
  font-size: .72rem !important;
  letter-spacing: .1em;
  color: var(--muted) !important;
  text-transform: uppercase;
}

/* ── chat input ───────────────────────────────────────────── */
[data-testid="stChatInput"] {
  background: var(--panel) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 4px !important;
  transition: border-color .25s ease;
}
[data-testid="stChatInput"]:focus-within { border-color: var(--teal) !important; }
[data-testid="stChatInput"] textarea { color: var(--paper) !important; font-size: .95rem !important; }
[data-testid="stBottomBlockContainer"] { background: var(--ink); }

/* ── sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] { background: #101315; border-right: 1px solid var(--rule); }
.side-h {
  font-family: 'Archivo', sans-serif;
  font-size: .68rem;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 1.6rem 0 .7rem;
}
.side-row {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  font-family: 'DM Mono', monospace;
  font-size: .72rem;
  padding: .3rem 0;
  border-bottom: 1px dotted var(--rule);
}
.side-row span:first-child { color: var(--muted); }
.side-row span:last-child  { color: var(--paper); text-align: right; }

/* ── empty state ──────────────────────────────────────────── */
.empty { margin-top: 1rem; }
.empty p {
  font-size: .92rem;
  color: var(--muted);
  line-height: 1.7;
  margin: 0 0 1.3rem;
}
.ex {
  font-family: 'DM Mono', monospace;
  font-size: .78rem;
  color: #9AA2A8;
  border-left: 1px solid var(--rule);
  padding: .1rem 0 .1rem .8rem;
  margin-bottom: .7rem;
}

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x if x is not None else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Brain — run rag.py's top half once and borrow its objects
# (script untouched — we just stop before its interactive input loop)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def boot():
    src = ENGINE_PATH.read_text(encoding="utf-8")
    src = re.split(r"^run\s*=\s*True\s*$", src, maxsplit=1, flags=re.M)[0]
    ns = {"__name__": "rag_engine", "__file__": str(ENGINE_PATH)}
    exec(compile(src, str(ENGINE_PATH), "exec"), ns)
    return ns


# ─────────────────────────────────────────────────────────────────────────────
# Query pipeline (mirror of rag.py's loop(); config pulled from rag.py via ns)
# ─────────────────────────────────────────────────────────────────────────────

def classify(ns, query):
    try:
        raw = ns["llm"].invoke(ns["prompt"] + f"user query: {query}").content.strip()
        return json.loads(raw)["appliance_name"]
    except Exception:
        return None


def retrieve(ns, query, appliance):
    dense = ns["embedder"].encode(
        "Represent this sentence for searching relevant passages: " + query,
        normalize_embeddings=True,
    )
    sparse = list(ns["sparse_embedder"].query_embed(query))[0].as_object()

    appliance_filter = None
    if appliance is not None:
        appliance_filter = Filter(
            must=[FieldCondition(key="appliance_name", match=MatchValue(value=appliance))]
        )

    out = []
    for entry in ns["files"]:
        file_filter = Filter(
            must=[FieldCondition(key="file_name", match=MatchValue(value=entry[1]))]
        )
        main_filter = Filter(
            must=(appliance_filter.must + file_filter.must)
            if appliance_filter is not None else file_filter.must
        )

        hits = ns["client"].query_points(
            collection_name="zenith_kb",
            prefetch=[
                Prefetch(query=dense, using="dense",
                         limit=ns["returned_chunks"], filter=main_filter),
                Prefetch(query=sparse, using="sparse",
                         limit=ns["returned_chunks"], filter=main_filter),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=ns["RFF_chunks"],
        )
        out.extend(hits.points)
    return out


def rerank(ns, query, points):
    if not points:
        return []
    pairs = [(query, p.payload["content"]) for p in points]
    scores = ns["ranker"].predict(pairs)
    scored = list(zip(points, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def format_chunks(scored, top_n):
    blocks = []
    for rank, (point, _) in enumerate(scored[:top_n], start=1):
        p = point.payload
        blocks.append(
            f"""[Source {rank}]
    Document: {p['file_name']}
    Section: {p['section']}
    Appliance: {p['appliance_name']}
    Content: {p['content'].strip()}
    """
        )
    return "\n".join(blocks)


def compose(ns, query, scored):
    top_n = ns["cross_encoder_chunks"]
    prompt1 = (
        f"system message:{ns['prompt_compile']}, query:{query}, "
        f"content:{format_chunks(scored, top_n)}"
    )
    return ns["llm"].invoke(prompt1).content


def parse_verdict(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return {
                "decision": data.get("decision"),
                "abstain_reason": data.get("abstain_reason"),
                "response": data.get("response", ""),
                "citations": data.get("citations") or [],
            }
        except Exception:
            pass
    return {"decision": None, "abstain_reason": None, "response": text, "citations": []}


# ─────────────────────────────────────────────────────────────────────────────
# Logging (mirror of the gspread block at the end of rag.py's loop())
# ─────────────────────────────────────────────────────────────────────────────

def log_to_sheet(query, verdict):
    gc = gspread.service_account(filename=LOG_CREDENTIALS)
    sh = gc.open_by_key(LOG_SHEET_KEY)
    worksheet = sh.worksheet(LOG_WORKSHEET)
    new_data = [
        query,
        verdict.get("decision"),
        verdict.get("abstain_reason"),
        verdict.get("response", ""),
        " ".join(verdict.get("citations") or []),
    ]
    worksheet.append_row(new_data)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

PLATE = {
    "answer":   ("answer",   "Answer"),
    "escalate": ("escalate", "Escalate to technician"),
    "abstain":  ("abstain",  "Not in knowledge base"),
}


def render_user(text):
    st.markdown(
        f'<div class="turn user"><div class="bubble">{esc(text)}</div></div>',
        unsafe_allow_html=True,
    )


def render_answer(msg):
    kind, label = PLATE.get(msg.get("decision"), ("abstain", "Response"))
    if msg.get("decision") == "abstain" and msg.get("abstain_reason") == "out_of_scope":
        label = "Out of scope"

    meta = []
    if msg.get("appliance"):
        meta.append(msg["appliance"])
    if msg.get("seconds"):
        meta.append(f"{msg['seconds']:.1f}s")
    meta_html = (
        f'<span class="plate-meta">{esc(" · ".join(meta))}</span>' if meta else ""
    )

    cites = "".join(f'<span class="cite">{esc(c)}</span>' for c in msg.get("citations", []))
    cites_html = f'<div class="cites">{cites}</div>' if cites else ""

    st.markdown(
        f"""<div class="turn bot">
  <div class="plate plate--{kind}">
    <span class="plate-dot"></span>
    <span class="plate-label">{esc(label)}</span>
    {meta_html}
  </div>
  <div class="response">{esc(msg.get("response", ""))}</div>
  {cites_html}
</div>""",
        unsafe_allow_html=True,
    )

    sources = msg.get("sources", [])
    if sources:
        with st.expander(f"Passages used — {len(sources)}"):
            for s in sources:
                st.markdown(
                    f"""<div class="src" style="animation-delay:{s['rank'] * 0.04:.2f}s">
  <div class="src-head">
    <span><b>{s['rank']:02d}</b> rank</span>
    <span>rerank {s['cross']:.3f}</span>
    <span>rrf {s['rrf']:.4f}</span>
    <span>{esc(s['file'])}</span>
    <span>{esc(s['type'])}</span>
  </div>
  <div class="src-sec">{esc(s['section'])}</div>
  <div class="src-body">{esc(s['content'])}</div>
</div>""",
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """<div class="mast">
  <h1>Zenith <span>◈</span> Field Service</h1>
  <p>Grounded answers for level-1 support</p>
  <div class="rule"></div>
</div>""",
    unsafe_allow_html=True,
)

boot_slot = st.empty()
if "booted" not in st.session_state:
    boot_slot.markdown(
        """<div class="work">
  <div class="work-label">building the index</div>
  <div class="work-track"><div class="work-fill" style="width:35%"></div></div>
</div>""",
        unsafe_allow_html=True,
    )

ns = boot()
st.session_state.booted = True
boot_slot.empty()

with st.sidebar:
    st.markdown('<div class="side-h">Index</div>', unsafe_allow_html=True)
    rows = [
        ("passages", ns["client"].count("zenith_kb").count),
        ("documents", len(ns["files"])),
        ("chunk size", f"{ns['chunk_size']} / {ns['chunk_overlap']}"),
    ]
    st.markdown('<div class="side-h">Retrieval</div>', unsafe_allow_html=True)
    rows += [
        ("dense", "bge-base-en-v1.5"),
        ("sparse", "bm25"),
        ("fusion", "RRF"),
        ("reranker", "ms-marco-MiniLM-L6"),
        ("per file", f"{ns['returned_chunks']} → {ns['RFF_chunks']}"),
        ("to model", ns["cross_encoder_chunks"]),
    ]
    st.markdown(
        "".join(
            f'<div class="side-row"><span>{esc(k)}</span><span>{esc(v)}</span></div>'
            for k, v in rows
        ),
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-h">Session</div>', unsafe_allow_html=True)
    if st.button("Clear transcript", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        """<div class="empty">
  <p>Type what the caller is describing. Every answer is drawn from the service
  manuals, fault-code tables, safety bulletins and past tickets — nothing else.</p>
  <div class="ex">What does error E3 mean on the IntelliCool split AC?</div>
  <div class="ex">The washer will not spin and the door stays locked.</div>
  <div class="ex">Customer smells gas near the outdoor unit.</div>
</div>""",
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    if msg["role"] == "user":
        render_user(msg["text"])
    else:
        render_answer(msg)

query = st.chat_input("Describe the caller's problem")

if query:
    render_user(query)
    st.session_state.messages.append({"role": "user", "text": query})

    slot = st.empty()

    def stage(label, pct):
        slot.markdown(
            f"""<div class="work">
  <div class="work-label">{esc(label)}</div>
  <div class="work-track"><div class="work-fill" style="width:{pct}%"></div></div>
</div>""",
            unsafe_allow_html=True,
        )

    started = time.time()
    verdict = None
    try:
        stage("identifying the appliance", 18)
        appliance = classify(ns, query)

        stage("searching the knowledge base", 48)
        points = retrieve(ns, query, appliance)

        stage("ranking the passages", 72)
        scored = rerank(ns, query, points)

        stage("writing the answer", 92)
        verdict = parse_verdict(compose(ns, query, scored))

        sources = [
            {
                "rank": i,
                "cross": float(score),
                "rrf": float(point.score),
                "file": point.payload["file_name"],
                "section": point.payload["section"],
                "type": point.payload["type"],
                "content": point.payload["content"].strip(),
            }
            for i, (point, score) in enumerate(
                scored[: ns["cross_encoder_chunks"]], start=1
            )
        ]

        message = {
            "role": "assistant",
            "appliance": appliance,
            "seconds": time.time() - started,
            "sources": sources,
            **verdict,
        }
    except Exception as exc:
        message = {
            "role": "assistant",
            "decision": "abstain",
            "abstain_reason": None,
            "response": f"The pipeline stopped before it could answer.\n\n{exc}",
            "citations": [],
            "sources": [],
            "seconds": time.time() - started,
        }

    slot.empty()
    render_answer(message)
    st.session_state.messages.append(message)

    # log after render so a slow or failed sheet write can't break the answer
    if verdict is not None:
        try:
            log_to_sheet(query, verdict)
        except Exception as log_exc:
            st.caption(f"⚠ could not log to sheet: {log_exc}")