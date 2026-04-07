import logging
from typing import List, Tuple
from datasets import load_dataset
from app.services.application.tag_service import TagService
from app.config import settings


logger = logging.getLogger(__name__)


def load_dbpedia_samples(split: str = "train", max_samples: int = 100) -> List[Tuple[str, str]]:
    """
    Load DBpedia samples from HuggingFace datasets.
    Returns list of (text, label_name) tuples.
    """
    try:
        logger.info(f"Loading DBpedia dataset (split={split}, max={max_samples})")
        ds = load_dataset("dbpedia_14", split=f"{split}[:{max_samples}]")
        label_names = ds.features["label"].names

        samples = []
        for item in ds:
            label_name = label_names[item["label"]]
            text = f"{item['title']}. {item['content']}"
            samples.append((text, label_name))

        logger.info(f"Loaded {len(samples)} DBpedia samples")
        return samples
    except Exception as e:
        logger.error(f"Failed to load DBpedia dataset: {e}")
        return []


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
