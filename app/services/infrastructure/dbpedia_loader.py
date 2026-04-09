import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import requests
from datasets import load_dataset
from app.services.application.tag_service import TagService
from app.config import settings


logger = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
SPARQL_HEADERS = {"Accept": "application/sparql-results+json"}


def _sparql_query(query: str, retries: int = 3, delay: float = 2.0) -> list[dict]:
    for attempt in range(retries):
        try:
            r = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=SPARQL_HEADERS,
                timeout=30,
            )
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"SPARQL attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(delay)
            else:
                raise


def load_dbpedia_ontology(cache_path: str | None = None) -> dict[str, str]:
    """
    Fetch all DBpedia ontology leaf classes with their English descriptions.
    Returns {ClassName: description_or_empty_string}.
    Results are cached to avoid repeated SPARQL calls.
    """
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading ontology from cache: {cache_path}")
        return json.loads(Path(cache_path).read_text())

    logger.info("Fetching DBpedia ontology from SPARQL endpoint...")
    query = """
        PREFIX owl:  <http://www.w3.org/2002/07/owl#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo:  <http://dbpedia.org/ontology/>

        SELECT DISTINCT ?label (SAMPLE(?comment) AS ?desc) WHERE {
          ?cls a owl:Class .
          ?cls rdfs:label ?label .
          FILTER(LANG(?label) = "en")
          FILTER(STRSTARTS(STR(?cls), "http://dbpedia.org/ontology/"))
          OPTIONAL {
            ?cls rdfs:comment ?comment
            FILTER(LANG(?comment) = "en")
          }
        }
        GROUP BY ?label
        ORDER BY ?label
    """
    bindings = _sparql_query(query)
    ontology = {
        b["label"]["value"]: b.get("desc", {}).get("value", "")
        for b in bindings
        if b.get("label", {}).get("value", "").strip()
    }
    logger.info(f"Loaded {len(ontology)} DBpedia ontology classes.")

    if cache_path:
        Path(cache_path).write_text(json.dumps(ontology, indent=2))
        logger.info(f"Ontology cached to {cache_path}")

    return ontology


def load_dbpedia_samples_sparql(
    ontology: dict[str, str],
    max_per_class: int = 50,
    min_length: int = 50,
    cache_path: str | None = None,
) -> List[Tuple[str, str]]:
    """
    Fetch Wikipedia abstracts for instances of each ontology class via SPARQL.
    Returns [(text, class_name), ...].
    """
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading samples from cache: {cache_path}")
        return [tuple(x) for x in json.loads(Path(cache_path).read_text())]

    samples = []
    for i, class_name in enumerate(ontology):
        query = f"""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?title ?abstract WHERE {{
              ?entity a dbo:{class_name} .
              ?entity rdfs:label ?title .
              ?entity dbo:abstract ?abstract .
              FILTER(LANG(?title) = "en")
              FILTER(LANG(?abstract) = "en")
              FILTER(STRLEN(?abstract) >= {min_length})
            }}
            LIMIT {max_per_class}
        """
        try:
            bindings = _sparql_query(query)
            for b in bindings:
                title = b.get("title", {}).get("value", "")
                abstract = b.get("abstract", {}).get("value", "")
                text = f"{title}. {abstract}".strip() if title else abstract
                if len(text) >= min_length:
                    samples.append((text, class_name))
            logger.info(
                f"[{i+1}/{len(ontology)}] {class_name}: {len(bindings)} samples"
            )
        except Exception as e:
            logger.warning(f"Skipping {class_name}: {e}")
        time.sleep(0.2)

    logger.info(f"Total samples: {len(samples)}")

    if cache_path:
        Path(cache_path).write_text(json.dumps(samples, indent=2))
        logger.info(f"Samples cached to {cache_path}")

    return samples


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
