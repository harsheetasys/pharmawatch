"""RAG over drug labels: "is this reaction already a KNOWN, labeled risk?"

Pipeline: label sections -> sentence-aware chunks (with overlap) -> embeddings
-> in-memory vector index (cosine) with metadata filtering by drug.

Embedders are pluggable:
* "tfidf"  - offline, zero-cost, deterministic (default; great for tests/CI)
* "gemini" - Gemini embedding model (semantic; set PW_EMBEDDER=gemini)

Known-vs-new is decided by HYBRID evidence: retrieve the most relevant chunks
for the reaction within that drug's label, then check them for the term
(spelling-normalized, e.g. MedDRA "haemorrhage" vs US-label "hemorrhage").
Dense retrieval alone is not trusted to say "yes, this is labeled".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import joblib
import numpy as np

from . import config, db

SECTION_RANK = {"boxed_warning": 0, "warnings_and_cautions": 1, "warnings": 1, "contraindications": 2,
                "adverse_reactions": 3}
STOP = {"of", "the", "and", "or", "in", "to", "a", "an", "with", "increased", "decreased", "disorder"}


# ---------------------------------------------------------------- chunking
def chunk_text(text: str, max_chars: int = 700, overlap_sentences: int = 1) -> list[str]:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    chunks, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s])) > max_chars:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ---------------------------------------------------------------- embedders
def _prep(t: str) -> str:
    return normalize_spelling(t.lower())


class TfidfEmbedder:
    name = "tfidf"

    def fit(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1,
                                   preprocessor=_prep)
        self.vec.fit(texts)
        return self

    def embed(self, texts, kind="document"):
        m = self.vec.transform(texts).toarray().astype(np.float32)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


class GeminiEmbedder:
    name = "gemini"

    def fit(self, texts):
        return self

    def embed(self, texts, kind="document"):
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        task = "RETRIEVAL_DOCUMENT" if kind == "document" else "RETRIEVAL_QUERY"
        out = []
        for i in range(0, len(texts), 100):
            res = client.models.embed_content(model=config.GEMINI_EMBED_MODEL, contents=texts[i:i + 100],
                                              config=types.EmbedContentConfig(task_type=task))
            out.extend(e.values for e in res.embeddings)
        m = np.array(out, dtype=np.float32)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

    def __getstate__(self):          # never pickle a client / key
        return {}


def make_embedder(kind: str | None = None):
    return GeminiEmbedder() if (kind or config.EMBEDDER) == "gemini" else TfidfEmbedder()


# ---------------------------------------------------------------- index
@dataclass
class Chunk:
    chunk_id: str
    drug: str
    section: str
    text: str


class LabelIndex:
    def __init__(self, chunks: list[Chunk], embedder):
        self.chunks = chunks
        self.embedder = embedder.fit([c.text for c in chunks])
        self.matrix = self.embedder.embed([c.text for c in chunks], kind="document")
        self.drugs = {c.drug.lower() for c in chunks}

    def search(self, query: str, drug: str | None = None, k: int = 4) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed([query], kind="query")[0]
        scores = self.matrix @ q
        idx = np.argsort(scores)[::-1]
        hits = []
        for i in idx:
            c = self.chunks[i]
            if drug and c.drug.lower() != drug.lower():
                continue
            hits.append((c, float(scores[i])))
            if len(hits) == k:
                break
        return hits


def build_index(engine=None, embedder=None) -> int:
    rows = db.query("""SELECT d.name AS drug, l.section, l.body
                       FROM label_sections l JOIN drugs d ON d.drug_id = l.drug_id
                       ORDER BY d.name, l.section""", engine=engine)
    chunks = []
    for r in rows.itertuples():
        for i, t in enumerate(chunk_text(r.body)):
            chunks.append(Chunk(f"{r.drug}:{r.section}:{i}", r.drug, r.section, t))
    index = LabelIndex(chunks, embedder or make_embedder())
    joblib.dump(index, config.INDEX_PATH)
    global _INDEX
    _INDEX = index
    return len(chunks)


_INDEX: LabelIndex | None = None


def load_index() -> LabelIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = joblib.load(config.INDEX_PATH)
    return _INDEX


# ---------------------------------------------------------------- known-risk check
_SPELL = [("haemorrhage", "hemorrhage"), ("oedema", "edema"), ("diarrhoea", "diarrhea"), ("anaemia", "anemia"),
          ("dyspnoea", "dyspnea"), ("aemia", "emia"), ("oesophag", "esophag"), ("haem", "hem"),
          ("paed", "ped"), ("oestr", "estr"), ("leukaemia", "leukemia")]


def normalize_spelling(text: str) -> str:
    for uk, us in _SPELL:
        text = text.replace(uk, us)
    return text


def _mentions(term: str, text: str) -> str | None:
    """Return a snippet around where the term is mentioned (exact phrase, or all content words
    in one sentence), else None. Snippet = ~150 chars before to ~250 after the match, so long
    bullet lists without full stops still show the RELEVANT part as the citation."""
    t = normalize_spelling(term.lower())
    words = [w for w in re.findall(r"[a-z]+", t) if w not in STOP]
    for sent in re.split(r"(?<=[.!?])\s+", text):
        s = normalize_spelling(sent.lower())
        if t in s:
            pos = s.find(t)
        elif len(words) > 1 and all(re.search(rf"\b{re.escape(w)}", s) for w in words):
            pos = re.search(rf"\b{re.escape(words[0])}", s).start()
        else:
            continue
        start = max(0, pos - 150)
        return ("..." if start else "") + sent[start:pos + 250].strip()
    return None

# Reporter / MedDRA wording -> wording labels often use (a tiny stand-in for the MedDRA hierarchy)
SYNONYMS = {
    "thyroid cancer": ["medullary thyroid carcinoma", "thyroid c-cell tumor"],
    "hair loss": ["alopecia"],
    "alopecia": ["hair loss"],
    "suicidal ideation": ["suicidal behavior and ideation", "suicidal thoughts"],
    "gastroparesis": ["delayed gastric emptying"],
    "impaired gastric emptying": ["delayed gastric emptying", "gastroparesis"],
    "injection site reaction": ["injection site reactions"],
}


def check_known_risk(drug: str, reactions: list[str], k: int = 4, index: LabelIndex | None = None) -> dict:
    """Hybrid check: scan EVERY chunk of this drug's label for the term (complete, exact),
    and use vector similarity only to rank evidence. Pure top-k retrieval missed real hits
    on long FDA labels (retriever recall failure), so lexical coverage comes first."""
    from .real_model import molecule_of
    drug = molecule_of(drug) or drug          # "Ozempic" -> "semaglutide" (labels are stored by molecule)
    index = index or load_index()
    if drug.lower() not in index.drugs:
        return {"drug": drug, "label_available": False,
                "results": [{"reaction": r, "status": "no_label_available", "evidence": []} for r in reactions]}
    drug_chunks = [c for c in index.chunks if c.drug.lower() == drug.lower()]
    results = []
    for r in reactions:
        variants = [r.lower()] + SYNONYMS.get(r.lower(), [])
        ranked = index.search(r, drug=drug, k=len(drug_chunks))           # similarity score for every chunk
        score_of = {c.chunk_id: s for c, s in ranked}
        evidence = []
        for c in drug_chunks:
            sent = next((m for v in variants if (m := _mentions(v, c.text))), None)
            if sent:
                evidence.append({"chunk_id": c.chunk_id, "section": c.section,
                                 "score": round(score_of.get(c.chunk_id, 0.0), 3), "quote": sent[:400]})
        evidence.sort(key=lambda e: (SECTION_RANK.get(e["section"], 9), -e["score"]))
        evidence = evidence[:k]
        best = evidence[0]["section"] if evidence else None
        if best is None:
            status = "not_in_label"
            evidence = [{"chunk_id": c.chunk_id, "section": c.section, "score": round(s, 3), "quote": c.text[:200]}
                        for c, s in ranked[:2]]   # show what WAS retrieved, for reviewer transparency
        elif best == "boxed_warning":
            status = "labeled_boxed_warning"
        elif best in ("warnings_and_cautions", "warnings", "contraindications"):
            status = "labeled_warning"
        else:
            status = "labeled_adverse_reaction"
        results.append({"reaction": r.lower(), "status": status, "evidence": evidence})
    return {"drug": drug, "label_available": True, "results": results}