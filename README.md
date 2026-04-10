# Semantic Tagging Service

A semantic tagging platform built on the **DBpedia Ontology Dataset**, using a **2-stage ML pipeline** for high-precision document tagging.

---

| Service | URL |
|---------|-----|
| Streamlit UI | http://95.216.169.118:8501 |
| FastAPI Docs | http://95.216.169.118:8000/docs |
| Qdrant Dashboard | http://95.216.169.118:6333/dashboard |

---


## ML Pipeline Details

### Bi-Encoder (Stage 1: Recall)
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dim: 384
- Storage: Qdrant (cosine similarity)
- Purpose: Fast ANN retrieval of top-K candidates

### Cross-Encoder (Stage 2: Precision)
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (base), fine-tuned on DBpedia ontology
- Input: `(document_text, tag_name)` pairs — tag names are normalized to lowercase readable form (e.g. `"NaturalPlace"` → `"natural place. A DBpedia category."`)
- Purpose:
  - Document tagging: score document against each candidate tag
  - UI suggestions: rerank similar tags

---

## Fine-tuning the Cross-Encoder

The cross-encoder was fine-tuned on the full DBpedia ontology (~380 classes). This is a broader version of the DBPedia dataset used on inference (14 classes). The data is downloaded from `https://downloads.dbpedia.org/2016-10/core-i18n/en/` and preprocessed to fit the format of (tag + description, document) pairs.



### Running fine-tuning script

```bash
uv run --group finetuning python -m app.finetuning.finetune_reranker \
  --train_samples 50 \
  --val_samples 10 \
  --epochs 3 \
  --batch_size 32 \
  --output_dir /models/cross_encoder_dbpedia
```


### Evaluate fine-tuning script

```bash
uv run --group finetuning python -m app.finetuning.evaluate \
  --max_per_class 20 \
  --finetuned_dir /models/cross_encoder_dbpedia
```


