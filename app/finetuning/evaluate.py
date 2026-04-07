"""
Evaluation: Baseline vs Pre-trained Cross-Encoder vs Fine-tuned Cross-Encoder
on DBpedia 14 test set.

Usage:
    python -m app.finetuning.evaluate [--n_samples 500]

Output:
    Console table + /tmp/eval_results.json
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

from app.config import settings, logger


def load_test_samples(n: int) -> List[Tuple[str, str]]:
    """Load n shuffled test samples so all 14 classes are represented."""
    logger.info(f"Loading {n} DBpedia test samples...")
    ds = load_dataset("dbpedia_14", split="test").shuffle(seed=42).select(range(n))
    label_names = ds.features["label"].names
    return [
        (f"{item['title']}. {item['content']}", label_names[item["label"]])
        for item in ds
    ]


def _fmt_tag(name: str, tag_format: str, descs: dict) -> str:
    if tag_format == "name+desc":
        desc = descs.get(name, "")
        return f"{name}: {desc}" if desc else name
    return name


def eval_baseline(samples: List[Tuple[str, str]]) -> Dict[str, float]:
    """Bi-encoder cosine similarity only — no cross-encoder."""
    logger.info("Evaluating: Baseline (bi-encoder cosine)...")
    enc = SentenceTransformer(settings.EMBEDDING_MODEL)

    tag_texts = [
        f"{n}: {settings.DBPEDIA_DESCRIPTIONS[n]}" for n in settings.DBPEDIA_CLASSES
    ]
    tag_embs = enc.encode(
        tag_texts, normalize_embeddings=True, show_progress_bar=False
    )

    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    correct = 0
    batch = 64

    for i in range(0, len(samples), batch):
        b = samples[i: i + batch]
        texts = [s[0][:400] for s in b]
        labels = [s[1] for s in b]
        doc_embs = enc.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        sims = doc_embs @ tag_embs.T
        for j, label in enumerate(labels):
            pred = settings.DBPEDIA_CLASSES[int(np.argmax(sims[j]))]
            per_class[label]["total"] += 1
            if pred == label:
                correct += 1
                per_class[label]["correct"] += 1
        if i % (batch * 10) == 0:
            logger.info(f"  {min(i + batch, len(samples))}/{len(samples)}")

    acc = correct / len(samples)
    return {
        "accuracy": round(acc, 4),
        "per_class": {
            c: round(
                per_class[c]["correct"] / max(per_class[c]["total"], 1), 4
            )
            for c in settings.DBPEDIA_CLASSES
        },
    }


def eval_cross_encoder(
    samples: List[Tuple[str, str]],
    model_path: str,
    tag_format: str,
    descs: dict,
    doc_max_chars: int,
    label: str,
) -> Dict[str, float]:
    """Evaluate a CrossEncoder model (pre-trained or fine-tuned)."""
    logger.info(f"Evaluating: {label} (model={model_path})...")
    model = CrossEncoder(model_path, max_length=256)

    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    correct = 0
    tag_texts = [_fmt_tag(t, tag_format, descs) for t in settings.DBPEDIA_CLASSES]

    t0 = time.time()
    for i, (doc_text, true_label) in enumerate(samples):
        doc = doc_text[:doc_max_chars]
        pairs = [[doc, t] for t in tag_texts]
        scores = model.predict(pairs, show_progress_bar=False)
        pred = settings.DBPEDIA_CLASSES[int(np.argmax(scores))]
        per_class[true_label]["total"] += 1
        if pred == true_label:
            correct += 1
            per_class[true_label]["correct"] += 1
        if i % 100 == 0 and i > 0:
            logger.info(f"  {i}/{len(samples)}  ({time.time() - t0:.1f}s)")

    acc = correct / len(samples)
    return {
        "accuracy": round(acc, 4),
        "per_class": {
            c: round(
                per_class[c]["correct"] / max(per_class[c]["total"], 1), 4
            )
            for c in settings.DBPEDIA_CLASSES
        },
    }


def run_evaluation(
    n_samples: int = 500,
    finetuned_dir: str = settings.FINETUNED_MODEL_DIR,
):
    samples = load_test_samples(n_samples)
    total = len(samples)
    results: dict = {"total_samples": total, "models": {}}

    # 1. Baseline
    results["models"]["baseline_biencoder"] = eval_baseline(samples)

    # 2. Pre-trained cross-encoder
    results["models"]["pretrained_crossencoder"] = eval_cross_encoder(
        samples,
        model_path=settings.RERANKER_BASE_MODEL,
        tag_format="name+desc",
        descs=settings.DBPEDIA_DESCRIPTIONS,
        doc_max_chars=400,
        label="Pretrained cross-encoder",
    )

    # 3. Fine-tuned cross-encoder (if available)
    ft_path = Path(finetuned_dir)
    if ft_path.exists() and any(ft_path.iterdir()):
        meta = {}
        meta_path = ft_path / "finetune_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        results["models"]["finetuned_crossencoder"] = eval_cross_encoder(
            samples,
            model_path=str(ft_path),
            tag_format=meta.get("tag_format", "name+desc"),
            descs=meta.get(
                "dbpedia_descriptions", settings.DBPEDIA_DESCRIPTIONS
            ),
            doc_max_chars=meta.get("doc_max_chars", 400),
            label="Fine-tuned cross-encoder",
        )
        results["finetune_meta"] = meta
    else:
        logger.warning(f"Fine-tuned model not found at {finetuned_dir}.")
        results["models"]["finetuned_crossencoder"] = None

    b_acc = results["models"]["baseline_biencoder"]["accuracy"]
    pt_acc = results["models"]["pretrained_crossencoder"]["accuracy"]
    ft_res = results["models"].get("finetuned_crossencoder")
    ft_acc = ft_res["accuracy"] if ft_res else None

    print("\n" + "=" * 72)
    print("EVALUATION RESULTS — DBpedia 14")
    print("=" * 72)
    print(f"Total samples : {total}")
    print()
    print(f"{'Model':<36}  {'Accuracy':>8}  {'vs Baseline':>12}")
    print(f"{'-'*36}  {'-'*8}  {'-'*12}")
    print(
        f"{'Baseline (bi-encoder cosine)':<36}  {b_acc:>8.1%}  {'—':>12}"
    )
    print(
        f"{'Pre-trained cross-encoder':<36}  "
        f"{pt_acc:>8.1%}  {(pt_acc - b_acc)*100:>+11.1f}pp"
    )
    if ft_acc is not None:
        print(
            f"{'Fine-tuned cross-encoder':<36}  "
            f"{ft_acc:>8.1%}  {(ft_acc - b_acc)*100:>+11.1f}pp"
        )
    print()
    print(
        f"{'Class':<28}  {'Baseline':>8}  {'Pretrained':>10}  "
        f"{'Finetuned':>10}"
    )
    print(f"{'-'*28}  {'-'*8}  {'-'*10}  {'-'*10}")
    for cls in settings.DBPEDIA_CLASSES:
        b = results["models"]["baseline_biencoder"]["per_class"][cls]
        pt = results["models"]["pretrained_crossencoder"]["per_class"][cls]
        ft_cls = ft_res["per_class"][cls] if ft_res else None
        ft_str = (
            f"{ft_cls:>9.1%}" if ft_cls is not None else f"{'N/A':>10}"
        )
        print(f"{cls:<28}  {b:>8.1%}  {pt:>10.1%}  {ft_str}")
    print("=" * 72)

    out_path = "/tmp/eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument(
        "--finetuned_dir",
        type=str,
        default=settings.FINETUNED_MODEL_DIR,
        help="Path to fine-tuned model directory",
    )
    args = p.parse_args()
    run_evaluation(args.n_samples, finetuned_dir=args.finetuned_dir)
