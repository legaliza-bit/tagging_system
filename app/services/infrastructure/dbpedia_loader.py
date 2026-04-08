import logging
from typing import List, Tuple
from datasets import load_dataset
from app.services.application.tag_service import TagService
from app.config import settings


logger = logging.getLogger(__name__)


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
    from app.db.repository import DocumentRepository, TagRepository
    from app.db.schemas import DocumentTag
    from app.services.infrastructure.embedding import EmbeddingService
    from app.services.infrastructure.vector_store import VectorStoreService

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

    # Build tag name -> tag object map
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
