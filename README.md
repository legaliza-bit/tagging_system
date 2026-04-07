# Semantic Tagging Service

A production-ready semantic tagging platform built on the **DBpedia Ontology Dataset**, using a **2-stage ML pipeline** for high-precision document tagging.

---

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI Docs | http://localhost:8000/docs |
| Qdrant Dashboard | http://localhost:6333/dashboard |

---


## ML Pipeline Details

### Bi-Encoder (Stage 1: Recall)
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dim: 384
- Storage: Qdrant (cosine similarity)
- Purpose: Fast ANN retrieval of top-K candidates

### Cross-Encoder (Stage 2: Precision)
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Input: `(document_text, tag_name)` pairs
- Purpose:
  - Document tagging: score document against each candidate tag
  - Tag deduplication: score new tag against existing tags
  - UI suggestions: rerank similar tags
