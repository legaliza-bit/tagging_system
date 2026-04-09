import argparse
import logging
import random
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset as HFDataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

from app.config import settings
from app.services.infrastructure.dbpedia_loader import (
    load_dbpedia_ontology,
    build_dataset,
    split_dataset,
    sample_per_class
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


@dataclass
class FinetuneConfig:
    base_model: str = settings.RERANKER_BASE_MODEL
    output_dir: str = settings.FINETUNED_MODEL_DIR
    train_samples: int = 20
    val_samples: int = 5
    hard_negatives: int = 3
    random_negatives: int = 1
    epochs: int = 3
    batch_size: int = 32
    max_length: int = 256
    warmup_ratio: float = 0.10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tag_format: str = "name+desc"
    doc_max_chars: int = 400


@dataclass
class Sample:
    id: int
    text: str
    label: str


def build_tag_texts(ontology, tag_format):
    def normalize(name):
        import re
        return re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower()

    tag_texts = {}
    for name, desc in ontology.items():
        readable = normalize(name)

        if desc:
            tag_texts[name] = f"{readable}. This is a type of {desc}"
        else:
            tag_texts[name] = f"{readable}. A DBpedia category."

    return tag_texts


def raw_to_samples(raw: list[tuple[str, str]], doc_max_chars: int) -> list[Sample]:
    return [
        Sample(id=i, text=text[:doc_max_chars], label=label)
        for i, (text, label) in enumerate(raw)
    ]


def mine_hard_negatives(samples, tag_texts, cfg):
    model = SentenceTransformer(settings.EMBEDDING_MODEL, device=cfg.device)

    tag_names = list(tag_texts.keys())
    tag_embs = model.encode(
        [tag_texts[t] for t in tag_names],
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=True,
    )

    texts = [s.text for s in samples]
    doc_embs = model.encode(
        texts, normalize_embeddings=True, batch_size=128, show_progress_bar=True
    )

    result = {}
    for idx, sample in enumerate(samples):
        sims = doc_embs[idx] @ tag_embs.T
        ranked = sorted(
            [
                (tag_names[j], float(sims[j]))
                for j in range(len(tag_names))
                if tag_names[j] != sample.label
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        top_k = 20
        candidates = ranked[:top_k]

        hard = [t for t, _ in candidates[:cfg.hard_negatives]]

        semi_pool = candidates[cfg.hard_negatives:]
        semi_hard = [random.choice(semi_pool)[0]] if semi_pool else []

        result[sample.id] = hard + semi_hard

    return result


def build_train_examples(samples, tag_texts, hard_neg_map, cfg) -> list[dict]:
    """Return list of {sentence1, sentence2, label} dicts for HFDataset."""
    rng = random.Random(cfg.seed)
    examples = []

    for s in samples:
        examples.append({"sentence1": s.text, "sentence2": tag_texts[s.label], "label": 1.0})
        for neg in hard_neg_map.get(s.id, []):
            examples.append({"sentence1": s.text, "sentence2": tag_texts[neg], "label": 0.0})
        other = [t for t in tag_texts if t != s.label]
        for neg in rng.sample(other, min(cfg.random_negatives, len(other))):
            examples.append({"sentence1": s.text, "sentence2": tag_texts[neg], "label": 0.0})

    rng.shuffle(examples)
    return examples


def build_val_examples(samples, tag_texts) -> list[dict]:
    NEG_VAL = 50
    return [
        {
            "query": s.text,
            "positive": [tag_texts[s.label]],
            "negative": random.sample(
                [tag_texts[t] for t in tag_texts if t != s.label],
                min(NEG_VAL, len(tag_texts)-1)
            )
        }
        for s in samples
    ]


def finetune(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    logger.info("Loading DBpedia ontology (~700 classes)...")
    ontology = load_dbpedia_ontology(cache_path=None)

    tag_texts = build_tag_texts(ontology, cfg.tag_format)

    logger.info("Loading dataset...")

    dataset = build_dataset(ontology, min_length=20)

    train, val = split_dataset(dataset, val_ratio=0.1)

    train_raw = sample_per_class(train, max_per_class=cfg.train_samples)
    val_raw = sample_per_class(val, max_per_class=cfg.train_samples)

    observed_labels = {label for _, label in train_raw}
    tag_texts = {k: v for k, v in tag_texts.items() if k in observed_labels}
    logger.info(f"Classes with training data: {len(tag_texts)}")

    train_samples = raw_to_samples(train_raw, cfg.doc_max_chars)
    val_samples = raw_to_samples(val_raw, cfg.doc_max_chars)

    logger.info(f"Train: {len(train_samples)} samples, Val: {len(val_samples)} samples")

    logger.info("Mining hard negatives...")
    hard_neg_map = mine_hard_negatives(train_samples, tag_texts, cfg)

    train_examples = build_train_examples(train_samples, tag_texts, hard_neg_map, cfg)
    val_examples = build_val_examples(val_samples, tag_texts)

    logger.info(f"Training pairs: {len(train_examples)}")

    model = CrossEncoder(
        cfg.base_model,
        num_labels=1,
        max_length=cfg.max_length,
        device=cfg.device,
    )

    train_dataset = HFDataset.from_list(train_examples)
    evaluator = CrossEncoderRerankingEvaluator(val_examples, name="val")

    training_args = CrossEncoderTrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        warmup_ratio=cfg.warmup_ratio,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=1.0,
        fp16=cfg.device == "cuda",
        seed=cfg.seed,
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        evaluator=evaluator,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    logger.info(f"Model saved to {cfg.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Fine-tune cross-encoder on full DBpedia ontology."
    )
    p.add_argument("--base_model",          default=settings.RERANKER_BASE_MODEL)
    p.add_argument("--output_dir",          default=settings.FINETUNED_MODEL_DIR)
    p.add_argument("--train_samples",       type=int,   default=50)
    p.add_argument("--val_samples",         type=int,   default=10)
    p.add_argument("--hard_negatives",      type=int,   default=3)
    p.add_argument("--random_negatives",    type=int,   default=1)
    p.add_argument("--epochs",              type=int,   default=3)
    p.add_argument("--batch_size",          type=int,   default=32)
    p.add_argument("--max_length",          type=int,   default=256)
    p.add_argument("--learning_rate",       type=float, default=2e-5)
    p.add_argument("--weight_decay",        type=float, default=0.01)
    p.add_argument("--warmup_ratio",        type=float, default=0.10)
    p.add_argument("--tag_format",          choices=["name", "name+desc"], default="name+desc")
    p.add_argument("--doc_max_chars",       type=int,   default=400)
    p.add_argument("--seed",                type=int,   default=42)
    finetune(FinetuneConfig(**vars(p.parse_args())))
