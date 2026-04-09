import bz2
import re
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
from datasets import load_dataset

from app.config import settings
from app.db.repository import DocumentRepository, TagRepository
from app.db.schemas import DocumentTag
from app.services.application.tag_service import TagService
from app.services.infrastructure.embedding import EmbeddingService
from app.services.infrastructure.vector_store import VectorStoreService


logger = logging.getLogger(__name__)


TRIPLE_RE = re.compile(r"<([^>]*)>\s+<([^>]*)>\s+<([^>]*)>")


def parse_instance_types(path):
    with bz2.open(path, "rt") as f:
        for line in f:
            if not line.startswith("<"):
                continue
            parts = line.split(" ", 3)
            if len(parts) < 3:
                continue

            s = parts[0].strip("<>")
            o = parts[2].strip("<>")

            yield s, o


def parse_abstracts(path):
    with bz2.open(path, "rt") as f:
        for line in f:
            if not line.startswith("<"):
                continue

            try:
                s, _, rest = line.split(" ", 2)
                s = s.strip("<>")

                start = rest.find('"')
                end = rest.rfind('"')

                if start == -1 or end == -1 or end <= start:
                    continue

                text = rest[start + 1:end]

                yield s, text

            except Exception:
                continue


def load_dbpedia_ontology(cache_path=None):
    """
    Returns {class_name: description}
    (description left empty to stay compatible)
    """
    if cache_path and Path(cache_path).exists():
        return json.loads(Path(cache_path).read_text())

    logger.info("Building ontology from instance types...")

    class_counts = defaultdict(int)

    for s, o in parse_instance_types(settings.INSTANCE_TYPES_PATH):
        BAD_CLASSES = {
            "Thing",
            "owl#Thing",
            "http://www.w3.org/2002/07/owl#Thing"
        }

        if o.split("/")[-1] in BAD_CLASSES:
            continue

        class_name = o.split("/")[-1]
        class_counts[class_name] += 1

    MIN_CLASS_SIZE = 50
    ontology = {
        cls: ""
        for cls, cnt in class_counts.items()
        if cnt >= MIN_CLASS_SIZE
    }

    logger.info(f"Ontology built: {len(ontology)} classes")

    if cache_path:
        Path(cache_path).write_text(json.dumps(ontology, indent=2))

    return ontology


def is_val_entity(entity_id, val_ratio):
    h = int(hashlib.md5(entity_id.encode()).hexdigest(), 16)
    return (h % 100) < int(val_ratio * 100)


def build_dataset(ontology, min_length):
    entity_to_class = {}

    for s, o in parse_instance_types(settings.INSTANCE_TYPES_PATH):
        cls = o.split("/")[-1]
        if cls in ontology and cls != "owl#Thing":
            entity_to_class[s] = cls

    dataset = []

    for s, text in parse_abstracts(settings.ABSTRACTS_PATH):
        if len(text) < min_length:
            continue

        cls = entity_to_class.get(s)
        if not cls:
            continue

        dataset.append((s, text, cls))

    return dataset


def split_dataset(dataset, val_ratio):
    train, val = [], []

    for s, text, cls in dataset:
        if is_val_entity(s, val_ratio):
            val.append((text, cls))
        else:
            train.append((text, cls))

    return train, val


def sample_per_class(data, max_per_class):
    counts = defaultdict(int)
    result = []

    for text, cls in data:
        if counts[cls] >= max_per_class:
            continue
        counts[cls] += 1
        result.append((text, cls))

    return result

# def load_dbpedia_samples_ft(
#     ontology,
#     max_per_class=50,
#     min_length=50,
#     cache_path=None,
#     split="train",
#     val_ratio=0.1,
# ):
#     if cache_path and Path(cache_path).exists():
#         return [tuple(x) for x in json.loads(Path(cache_path).read_text())]

#     logger.info("Building samples (optimized streaming version)...")

#     entity_to_class = {}

#     # PASS 1: entity → class (streaming, cheap)
#     for s, o in parse_instance_types(settings.INSTANCE_TYPES_PATH):
#         cls = o.split("/")[-1]

#         if cls in ontology and cls != "owl#Thing":
#             entity_to_class[s] = cls

#     class_counts_train = defaultdict(int)
#     class_counts_val = defaultdict(int)

#     samples_train = []
#     samples_val = []

#     for s, text in parse_abstracts(settings.ABSTRACTS_PATH):

#         if len(text) < min_length:
#             continue

#         cls = entity_to_class.get(s)
#         if not cls:
#             continue

#         is_val = stable_bucket(s) < int(val_ratio * 100)

#         if is_val:
#             if class_counts_val[cls] >= max_per_class:
#                 continue
#             class_counts_val[cls] += 1
#             samples_val.append((text, cls))
#         else:
#             if class_counts_train[cls] >= max_per_class:
#                 continue
#             class_counts_train[cls] += 1
#             samples_train.append((text, cls))

#     logger.info(f"train samples: {len(samples_train)}")
#     logger.info(f"val samples: {len(samples_val)}")

#     result = samples_train if split == "train" else samples_val

#     if cache_path:
#         Path(cache_path).write_text(json.dumps(result, indent=2))

#     return result


def load_dbpedia_samples(
    split: str = "train",
    max_per_class: int = 100,
    min_length: int = 50,
) -> List[Tuple[str, str]]:

    ds = load_dataset("dbpedia_14", split=split)
    ds = ds.shuffle(seed=42)

    label_names = ds.features["label"].names
    per_class = {name: [] for name in label_names}
    filled = set()

    for item in ds:
        label = label_names[item["label"]]

        if label in filled:
            continue

        text = f"{item['title']}. {item['content']}".strip()

        if len(text) < min_length:
            continue

        per_class[label].append(text)

        if len(per_class[label]) >= max_per_class:
            filled.add(label)

        if len(filled) == len(label_names):
            break

    samples = [
        (text, label)
        for label, texts in per_class.items()
        for text in texts
    ]
    logger.info(f"Loaded {len(samples)} DBPedia samples, split={split}")
    return samples


async def seed_dbpedia_tags(db_session):
    """Seed all 14 DBpedia classes as tags."""
    svc = TagService(db_session)

    logger.info("Seeding DBpedia tags...")
    created = 0
    for class_name in settings.DBPEDIA_CLASSES:
        desc = settings.DBPEDIA_DESCRIPTIONS.get(class_name, "")
        result = await svc.create_tag(
            name=class_name,
            description=desc,
            force_create=False,
            source="dbpedia",
        )
        if result.created:
            created += 1

    logger.info(f"Seeded {created} new DBpedia tags ({len(settings.DBPEDIA_CLASSES)} total)")
    return created


async def seed_dbpedia_documents(db_session, max_per_class: int = 10):
    """Seed a sample of DBpedia documents with ground-truth tag assignments."""

    samples = load_dbpedia_samples(split="test", max_per_class=max_per_class)
    if not samples:
        logger.warning("No DBpedia samples loaded, skipping document seeding")
        return 0

    per_class: dict[str, list] = {name: [] for name in settings.DBPEDIA_CLASSES}
    for text, label in samples:
        if len(per_class[label]) < max_per_class:
            per_class[label].append(text)

    doc_repo = DocumentRepository(db_session)
    tag_repo = TagRepository(db_session)
    embedder = EmbeddingService.get_instance()
    vector_store = VectorStoreService.get_instance()

    tag_map = {}
    for name in settings.DBPEDIA_CLASSES:
        tag = await tag_repo.get_by_name(name)
        if tag:
            tag_map[name] = tag

    created = 0
    for label, texts in per_class.items():
        tag = tag_map.get(label)
        if not tag:
            continue
        for text in texts:
            doc = await doc_repo.create(content=text, title=label, dbpedia_label=label)
            embedding = embedder.embed_one(text[:500])
            await vector_store.upsert_document(doc.id, embedding, content_snippet=text[:200])
            dt = DocumentTag(document_id=doc.id, tag_id=tag.id, confidence=1.0, is_human_verified=True)
            db_session.add(dt)
            await db_session.flush()
            created += 1

    await db_session.commit()
    logger.info(f"Seeded {created} DBpedia documents")
    return created
