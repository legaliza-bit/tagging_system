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
- Input: (tag + description, document) pairs — tag names are normalized to lowercase readable form (e.g. `"NaturalPlace"` → `"natural place. A DBpedia category."`)
- Purpose:
  - Document tagging: score document against each candidate tag
  - UI suggestions: rerank similar tags

---

## Fine-tuning the Cross-Encoder

The cross-encoder was fine-tuned on the full DBpedia ontology (~380 classes). This is a broader version of the DBPedia dataset used on inference (14 classes). The data is downloaded from `https://downloads.dbpedia.org/2016-10/core-i18n/en/` and preprocessed to fit the format of (tag + description, document) pairs. Each tag receives 1 positive example, 3 hard-negatives and 1 soft-negative. The hard negatives are appointed by the bi-encoder model based on cosine distance to the tag, and the soft negatives are purely random.

The cross-encoder is trained using BinaryCrossEntropyLoss. Positive pairs (tag + description, document) are appointed a score of 1.0, negatives -- a score of 0.0.


### Running fine-tuning script

```bash
uv run --group finetuning python -m app.finetuning.finetune_reranker \
  --train_samples 50 \
  --val_samples 10 \
  --epochs 3 \
  --batch_size 32 \
  --output_dir /models/cross_encoder_dbpedia
```


### Evaluating

```bash
uv run --group finetuning python -m app.finetuning.evaluate \
  --max_per_class 20 \
  --finetuned_dir /models/cross_encoder_dbpedia
```

### Eval metrics

Total samples  : 7052
Classes tested : 369

Model                                 Accuracy   vs Baseline
------------------------------------  --------  ------------
Baseline (bi-encoder cosine)             53.2%             —
Pre-trained cross-encoder                39.3%        -13.9pp
Fine-tuned cross-encoder                 74.3%        +21.0pp

Class                               Baseline  Pretrained   Finetuned
-------------------------------------------------------------------
  -- Top 15 classes --
EurovisionSongContestEntry            100.0%       95.0%      100.0%
Cyclist                                35.0%       85.0%      100.0%
SumoWrestler                          100.0%      100.0%      100.0%
SkiArea                                90.0%       75.0%      100.0%
Baronet                                90.0%       80.0%      100.0%
LaunchPad                              60.0%       70.0%      100.0%
ComedyGroup                            66.7%      100.0%      100.0%
Grape                                  30.0%       40.0%      100.0%
Diocese                               100.0%       75.0%      100.0%
ClassicalMusicArtist                   60.0%       15.0%      100.0%
Jockey                                 85.0%       65.0%      100.0%
LawFirm                               100.0%       85.0%      100.0%
SquashPlayer                           90.0%       90.0%      100.0%
Arachnid                               35.0%       25.0%      100.0%
Library                                40.0%       30.0%      100.0%
  -- Bottom 15 classes --
SportsTeam                             10.0%       15.0%        0.0%
SportsEvent                             5.0%        0.0%        0.0%
HistoricPlace                           5.0%        0.0%        0.0%
YearInSpaceflight                      50.0%       50.0%        0.0%
Venue                                   0.0%        0.0%        0.0%
TelevisionHost                         28.6%       14.3%        0.0%
MilitaryStructure                       0.0%       10.0%        0.0%
Stadium                                85.0%        5.0%        0.0%
HistoricBuilding                        5.0%        0.0%        0.0%
MusicFestival                          90.0%       55.0%        0.0%
Place                                   0.0%        5.0%        0.0%
MilitaryPerson                         65.0%        5.0%        0.0%
Building                                0.0%        0.0%        0.0%
City                                   20.0%       10.0%        0.0%
Politician                              5.0%       10.0%        0.0%
========================================================================

So, as you can see, the finetuned model has learnt to distinguish the more frequent classes better. The rare classes are missing from the training data on purpose, so the bottom classes inevitably score worse.