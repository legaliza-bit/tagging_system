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
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
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


@dataclass
class Sample:
    id: int
    text: str
    label: str


def format_tag(name: str, tag_format: str) -> str:
    if tag_format == "name+desc":
        desc = settings.DBPEDIA_DESCRIPTIONS.get(name, "")
        return f"{name}: {desc}" if desc else name
    return name


def build_tag_texts(tag_format: str):
    return {
        t: format_tag(t, tag_format)
        for t in settings.DBPEDIA_CLASSES
    }


def load_samples(split: str, n: int, doc_max_chars: int) -> list[Sample]:
    raw = load_dbpedia_samples(split, n)
    return [
        Sample(
            id=i,
            text=text[:doc_max_chars],
            label=label
        )
        for i, (text, label) in enumerate(raw)
    ]


def mine_hard_negatives(samples, tag_texts, cfg):
    model = SentenceTransformer(settings.EMBEDDING_MODEL, device=cfg.device)

    tag_names = list(tag_texts.keys())
    tag_embs = model.encode(
        [tag_texts[t] for t in tag_names],
        normalize_embeddings=True
    )

    result = {}

    for sample in samples:
        doc_emb = model.encode(sample.text, normalize_embeddings=True)
        sims = doc_emb @ tag_embs.T

        ranked = sorted(
            [
                (tag_names[i], float(sims[i]))
                for i in range(len(tag_names))
                if tag_names[i] != sample.label
            ],
            key=lambda x: x[1],
            reverse=True
        )

        result[sample.id] = [t for t, _ in ranked[:cfg.hard_negatives]]

    return result


def build_train_examples(samples, tag_texts, hard_neg_map, cfg):
    rng = random.Random(cfg.seed)
    examples = []

    for s in samples:
        # positive
        examples.append(InputExample(
            texts=[s.text, tag_texts[s.label]],
            label=1.0
        ))

        # hard negatives
        for neg in hard_neg_map.get(s.id, []):
            examples.append(InputExample(
                texts=[s.text, tag_texts[neg]],
                label=0.0
            ))

        # random negatives
        other = [t for t in tag_texts if t != s.label]
        for neg in rng.sample(other, min(cfg.random_negatives, len(other))):
            examples.append(InputExample(
                texts=[s.text, tag_texts[neg]],
                label=0.0
            ))

    rng.shuffle(examples)
    return examples


def build_val_examples(samples, tag_texts):
    return [
        {
            "query": s.text,
            "positive": [tag_texts[s.label]],
            "negative": [
                tag_texts[t]
                for t in tag_texts if t != s.label
            ],
        }
        for s in samples
    ]


def train(cfg):
    train_samples = load_samples("train", cfg.train_samples, cfg.doc_max_chars)
    val_samples = load_samples("test", cfg.val_samples, cfg.doc_max_chars)

    tag_texts = build_tag_texts(cfg.tag_format)

    hard_neg_map = mine_hard_negatives(train_samples, tag_texts, cfg)

    train_examples = build_train_examples(
        train_samples, tag_texts, hard_neg_map, cfg
    )

    val_examples = build_val_examples(val_samples, tag_texts)

    model = CrossEncoder(
        cfg.base_model,
        num_labels=1,
        max_length=cfg.max_length,
        device=cfg.device
    )

    loader = DataLoader(train_examples, batch_size=cfg.batch_size, shuffle=True)

    evaluator = CERerankingEvaluator(val_examples)

    model.fit(
        train_dataloader=loader,
        evaluator=evaluator,
        epochs=cfg.epochs,
        warmup_steps=int(len(loader) * cfg.epochs * cfg.warmup_ratio),
        optimizer_params={"lr": cfg.learning_rate},
        max_grad_norm=1.0,
        output_path=cfg.output_dir,
    )


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
    train(FinetuneConfig(**vars(p.parse_args())))
