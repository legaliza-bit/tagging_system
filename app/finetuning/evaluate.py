"""
Evaluation: Baseline vs Pre-trained Cross-Encoder vs Fine-tuned Cross-Encoder
on DBpedia ontology (local TTL dump).

Usage:
    python -m app.finetuning.evaluate [options]

Output:
    Console summary + /tmp/eval_results.json
"""
import argparse
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

from app.config import settings
from app.services.infrastructure.dbpedia_loader import (
    load_dbpedia_ontology,
    build_dataset,
    split_dataset,
    sample_per_class,
)


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")


def build_tag_texts(ontology: dict[str, str]) -> dict[str, str]:
    """Match the inference format in reranker._fmt_tag."""
    return {
        name: f"{re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower()}. A DBpedia category."
        for name in ontology
    }


def eval_baseline(
    samples: List[Tuple[str, str]],
    tag_texts: dict[str, str],
    doc_max_chars: int,
) -> Dict:
    """Bi-encoder cosine similarity only — no cross-encoder."""
    logger.info("Evaluating: Baseline (bi-encoder cosine)...")
    enc = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)

    tag_names = list(tag_texts.keys())
    tag_embs = enc.encode(
        [tag_texts[t] for t in tag_names],
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=True,
    )

    per_class: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    correct = 0
    batch = 64

    for i in range(0, len(samples), batch):
        b = samples[i: i + batch]
        texts = [s[0][:doc_max_chars] for s in b]
        labels = [s[1] for s in b]
        doc_embs = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        sims = doc_embs @ tag_embs.T
        for j, label in enumerate(labels):
            pred = tag_names[int(np.argmax(sims[j]))]
            per_class[label]["total"] += 1
            if pred == label:
                correct += 1
                per_class[label]["correct"] += 1
        if i % (batch * 10) == 0 and i > 0:
            logger.info(f"  {min(i + batch, len(samples))}/{len(samples)}")

    acc = correct / len(samples)
    return {
        "accuracy": round(acc, 4),
        "per_class": {
            c: round(v["correct"] / max(v["total"], 1), 4)
            for c, v in per_class.items()
        },
    }


def eval_cross_encoder(
    samples: List[Tuple[str, str]],
    tag_texts: dict[str, str],
    model_path: str,
    doc_max_chars: int,
    label: str,
) -> Dict:
    """Evaluate a CrossEncoder model (pre-trained or fine-tuned)."""
    logger.info(f"Evaluating: {label} (model={model_path})...")
    model = CrossEncoder(model_path, max_length=256, device=device)

    tag_names = list(tag_texts.keys())
    tag_text_list = [tag_texts[t] for t in tag_names]

    per_class: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    correct = 0
    t0 = time.time()

    for i, (doc_text, true_label) in enumerate(samples):
        doc = doc_text[:doc_max_chars]
        pairs = [[doc, t] for t in tag_text_list]
        scores = model.predict(pairs, show_progress_bar=False)
        pred = tag_names[int(np.argmax(scores))]
        per_class[true_label]["total"] += 1
        if pred == true_label:
            correct += 1
            per_class[true_label]["correct"] += 1
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - t0
            logger.info(f"  {i}/{len(samples)}  ({elapsed:.1f}s)")

    acc = correct / len(samples)
    return {
        "accuracy": round(acc, 4),
        "per_class": {
            c: round(v["correct"] / max(v["total"], 1), 4)
            for c, v in per_class.items()
        },
    }


def _print_per_class_table(results: dict, top_n: int = 15):
    model_keys = [k for k, v in results["models"].items() if v is not None]
    if not model_keys:
        return

    all_classes: set = set()
    for k in model_keys:
        all_classes.update(results["models"][k].get("per_class", {}).keys())

    if not all_classes:
        return

    sort_key = (
        "finetuned_crossencoder"
        if "finetuned_crossencoder" in model_keys
        else "pretrained_crossencoder"
    )
    ranked = sorted(
        all_classes,
        key=lambda c: results["models"][sort_key].get("per_class", {}).get(c, 0.0),
        reverse=True,
    )

    short_names = {
        "baseline_biencoder": "Baseline",
        "pretrained_crossencoder": "Pretrained",
        "finetuned_crossencoder": "Finetuned",
    }
    header = f"{'Class':<32}" + "".join(
        f"  {short_names.get(k, k[:10]):>10}" for k in model_keys
    )
    print(header)
    print("-" * (32 + 12 * len(model_keys)))

    def print_row(cls):
        row = f"{cls:<32}"
        for k in model_keys:
            v = results["models"][k].get("per_class", {}).get(cls)
            row += f"  {v:>10.1%}" if v is not None else f"  {'N/A':>10}"
        print(row)

    print(f"  -- Top {top_n} classes --")
    for cls in ranked[:top_n]:
        print_row(cls)
    print(f"  -- Bottom {top_n} classes --")
    for cls in ranked[-top_n:]:
        print_row(cls)


def run_evaluation(
    max_per_class: int = 20,
    doc_max_chars: int = 400,
    finetuned_dir: str = settings.FINETUNED_MODEL_DIR,
    ontology_cache: str | None = None,
    top_n: int = 15,
):
    logger.info("Loading DBpedia ontology...")
    ontology = load_dbpedia_ontology(cache_path=ontology_cache)
    logger.info(f"Ontology: {len(ontology)} classes")

    logger.info("Loading evaluation dataset...")
    dataset = build_dataset(ontology, min_length=20)
    _, val_raw = split_dataset(dataset, val_ratio=0.1)
    val_raw = sample_per_class(val_raw, max_per_class=max_per_class)

    observed = {label for _, label in val_raw}
    ontology = {k: v for k, v in ontology.items() if k in observed}
    logger.info(f"Classes with evaluation data: {len(ontology)}, samples: {len(val_raw)}")

    tag_texts = build_tag_texts(ontology)
    samples: List[Tuple[str, str]] = [(text[:doc_max_chars], label) for text, label in val_raw]

    results: dict = {
        "total_samples": len(samples),
        "ontology_size": len(ontology),
        "models": {},
    }

    results["models"]["baseline_biencoder"] = eval_baseline(
        samples, tag_texts, doc_max_chars=doc_max_chars
    )

    results["models"]["pretrained_crossencoder"] = eval_cross_encoder(
        samples,
        tag_texts=tag_texts,
        model_path=settings.RERANKER_BASE_MODEL,
        doc_max_chars=doc_max_chars,
        label="Pretrained cross-encoder",
    )

    ft_path = Path(finetuned_dir)
    if ft_path.exists() and any(ft_path.iterdir()):
        results["models"]["finetuned_crossencoder"] = eval_cross_encoder(
            samples,
            tag_texts=tag_texts,
            model_path=str(ft_path),
            doc_max_chars=doc_max_chars,
            label="Fine-tuned cross-encoder",
        )
    else:
        logger.warning(f"Fine-tuned model not found at {finetuned_dir}.")
        results["models"]["finetuned_crossencoder"] = None

    b_acc = results["models"]["baseline_biencoder"]["accuracy"]
    pt_acc = results["models"]["pretrained_crossencoder"]["accuracy"]
    ft_res = results["models"].get("finetuned_crossencoder")
    ft_acc = ft_res["accuracy"] if ft_res else None

    print("\n" + "=" * 72)
    print("EVALUATION RESULTS — DBpedia ontology (local TTL)")
    print("=" * 72)
    print(f"Total samples  : {len(samples)}")
    print(f"Classes tested : {len(ontology)}")
    print()
    print(f"{'Model':<36}  {'Accuracy':>8}  {'vs Baseline':>12}")
    print(f"{'-'*36}  {'-'*8}  {'-'*12}")
    print(f"{'Baseline (bi-encoder cosine)':<36}  {b_acc:>8.1%}  {'—':>12}")
    print(f"{'Pre-trained cross-encoder':<36}  {pt_acc:>8.1%}  {(pt_acc - b_acc)*100:>+11.1f}pp")
    if ft_acc is not None:
        print(f"{'Fine-tuned cross-encoder':<36}  {ft_acc:>8.1%}  {(ft_acc - b_acc)*100:>+11.1f}pp")
    print()
    _print_per_class_table(results, top_n=top_n)
    print("=" * 72)

    out_path = "/tmp/eval_results.json"
    Path(out_path).write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved → {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate cross-encoder on DBpedia ontology (local TTL dump)."
    )
    p.add_argument("--max_per_class",  type=int, default=20,
                   help="Max evaluation samples per class from the val split")
    p.add_argument("--doc_max_chars",  type=int, default=400)
    p.add_argument("--finetuned_dir",  default=settings.FINETUNED_MODEL_DIR)
    p.add_argument("--ontology_cache", default=None,
                   help="Path to cache the ontology JSON (optional)")
    p.add_argument("--top_n",          type=int, default=15,
                   help="Number of best/worst classes to show in per-class table")
    args = p.parse_args()
    run_evaluation(
        max_per_class=args.max_per_class,
        doc_max_chars=args.doc_max_chars,
        finetuned_dir=args.finetuned_dir,
        ontology_cache=args.ontology_cache,
        top_n=args.top_n,
    )
