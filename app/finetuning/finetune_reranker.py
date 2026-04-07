import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader

from app.config import settings, logger
from app.services.infrastructure.dbpedia_loader import load_dbpedia_samples


@dataclass
class FinetuneConfig:
    base_model: str = settings.RERANKER_BASE_MODEL
    output_dir: str = settings.FINETUNED_MODEL_DIR
    train_samples: int = 5_000
    val_samples: int = 500
    hard_negatives: int = 3
    random_negatives: int = 1
    epochs: int = 3
    batch_size: int = 32
    max_length: int = 256
    warmup_ratio: float = 0.10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    seed: int = 42
    device: str = "cpu"
    tag_format: str = "name+desc"
    doc_max_chars: int = 400


def format_tag(name: str, tag_format: str) -> str:
    if tag_format == "name+desc":
        desc = settings.DBPEDIA_DESCRIPTIONS.get(name, "")
        return f"{name}: {desc}" if desc else name
    return name


def mine_hard_negatives(
    samples: List[Tuple[str, str]],
    n_hard: int,
    doc_max_chars: int,
    batch_size: int = 128,
) -> Dict[str, List[str]]:
    """Use bi-encoder to find the top-n wrong tags most likely to confuse the model."""
    logger.info(f"Mining hard negatives (n_hard={n_hard})...")
    biencoder = SentenceTransformer(settings.EMBEDDING_MODEL)
    tag_texts = [format_tag(t, "name+desc") for t in settings.DBPEDIA_CLASSES]
    tag_embs = biencoder.encode(tag_texts, normalize_embeddings=True, show_progress_bar=False)

    texts = [s[0][:doc_max_chars] for s in samples]
    labels = [s[1] for s in samples]
    result: Dict[str, List[str]] = {}

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        batch_labels = labels[i: i + batch_size]
        doc_embs = biencoder.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
        sims = doc_embs @ tag_embs.T

        for j, (sim_row, true_label) in enumerate(zip(sims, batch_labels)):
            ranked = sorted(
                [(settings.DBPEDIA_CLASSES[k], float(sim_row[k]))
                 for k in range(len(settings.DBPEDIA_CLASSES))
                 if settings.DBPEDIA_CLASSES[k] != true_label],
                key=lambda x: x[1], reverse=True,
            )
            result[batch_texts[j]] = [t for t, _ in ranked[:n_hard]]

        if i % (batch_size * 8) == 0:
            logger.info(f"  {min(i + batch_size, len(texts)):,}/{len(texts):,}")

    return result


def build_train_examples(
    samples: List[Tuple[str, str]],
    cfg: FinetuneConfig,
    hard_neg_map: Dict[str, List[str]],
) -> List[InputExample]:
    rng = random.Random(cfg.seed)
    examples: List[InputExample] = []

    for doc_text, true_label in samples:
        doc = doc_text[:cfg.doc_max_chars]
        examples.append(InputExample(texts=[doc, format_tag(true_label, cfg.tag_format)], label=1.0))
        for neg in hard_neg_map.get(doc, [])[:cfg.hard_negatives]:
            examples.append(InputExample(texts=[doc, format_tag(neg, cfg.tag_format)], label=0.0))
        other = [t for t in settings.DBPEDIA_CLASSES if t != true_label]
        for neg in rng.sample(other, min(cfg.random_negatives, len(other))):
            examples.append(InputExample(texts=[doc, format_tag(neg, cfg.tag_format)], label=0.0))

    rng.shuffle(examples)
    logger.info(f"Training set: {len(examples):,} examples.")
    return examples


def build_val_examples(samples: List[Tuple[str, str]], cfg: FinetuneConfig) -> List[InputExample]:
    """Full cross-product: every (doc, tag) pair, label=1 iff correct."""
    examples = []
    for doc_text, true_label in samples:
        doc = doc_text[:cfg.doc_max_chars]
        for tag_name in settings.DBPEDIA_CLASSES:
            examples.append(InputExample(
                texts=[doc, format_tag(tag_name, cfg.tag_format)],
                label=1.0 if tag_name == true_label else 0.0,
            ))
    return examples


def finetune(cfg: FinetuneConfig) -> str:
    """Run the full fine-tuning pipeline. Returns saved model directory."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    logger.info(f"Device: {device}")
    logger.info(f"Config:\n{json.dumps(asdict(cfg), indent=2)}")

    train_raw = load_dbpedia_samples("train", cfg.train_samples)
    val_raw = load_dbpedia_samples("test",  cfg.val_samples)

    hard_neg_map = mine_hard_negatives(train_raw, n_hard=cfg.hard_negatives, doc_max_chars=cfg.doc_max_chars)
    train_examples = build_train_examples(train_raw, cfg, hard_neg_map)
    val_examples = build_val_examples(val_raw, cfg)

    model = CrossEncoder(cfg.base_model, num_labels=1, max_length=cfg.max_length, device=device)

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=cfg.batch_size)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_examples, name="dbpedia_val")
    total_steps = len(train_loader) * cfg.epochs

    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        epochs=cfg.epochs,
        warmup_steps=int(total_steps * cfg.warmup_ratio),
        output_path=str(output_dir),
        save_best_model=True,
        optimizer_params={"lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
        show_progress_bar=True,
        evaluation_steps=max(1, len(train_loader) // 4),
    )

    correct = sum(
        settings.DBPEDIA_CLASSES[int(np.argmax(
            model.predict([[doc_text[:cfg.doc_max_chars], format_tag(t, cfg.tag_format)]
                           for t in settings.DBPEDIA_CLASSES], show_progress_bar=False)
        ))] == true_label
        for doc_text, true_label in val_raw
    )
    final_acc = correct / len(val_raw)
    logger.info(f"Final top-1 accuracy: {final_acc:.1%}")

    meta = {
        **asdict(cfg),
        "final_top1_accuracy": round(final_acc, 4),
        "dbpedia_classes": settings.DBPEDIA_CLASSES,
        "dbpedia_descriptions": settings.DBPEDIA_DESCRIPTIONS,
    }
    with open(output_dir / "finetune_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nFine-tuned model: {output_dir}  |  top-1 acc: {final_acc:.1%}\n")
    return str(output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune cross-encoder on DBpedia.")
    p.add_argument("--base_model",       default=settings.RERANKER_BASE_MODEL)
    p.add_argument("--output_dir",       default=settings.FINETUNED_MODEL_DIR)
    p.add_argument("--train_samples",    type=int,   default=5_000)
    p.add_argument("--val_samples",      type=int,   default=500)
    p.add_argument("--hard_negatives",   type=int,   default=3)
    p.add_argument("--random_negatives", type=int,   default=1)
    p.add_argument("--epochs",           type=int,   default=3)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--max_length",       type=int,   default=256)
    p.add_argument("--learning_rate",    type=float, default=2e-5)
    p.add_argument("--weight_decay",     type=float, default=0.01)
    p.add_argument("--warmup_ratio",     type=float, default=0.10)
    p.add_argument("--tag_format",       choices=["name", "name+desc"], default="name+desc")
    p.add_argument("--doc_max_chars",    type=int,   default=400)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "cuda", "mps"])
    finetune(FinetuneConfig(**vars(p.parse_args())))
