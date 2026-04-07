import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import documents, tags
from app.db.schemas import AsyncSessionLocal, init_db
from app.db.repository import TagRepository
from app.services.infrastructure.embedding import EmbeddingService
from app.services.infrastructure.reranker import RerankerService
from app.services.infrastructure.vector_store import VectorStoreService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Semantic Tagging Service...")

    await init_db()
    logger.info("PostgreSQL schema initialized.")

    vs = VectorStoreService.get_instance()
    await vs.ensure_collections()
    logger.info("Qdrant collections ready.")

    EmbeddingService.get_instance()
    RerankerService.get_instance()
    logger.info("ML models loaded.")

    try:
        from app.services.infrastructure.dbpedia_loader import seed_dbpedia_tags
        async with AsyncSessionLocal() as session:
            repo = TagRepository(session)
            count = await repo.count()
            if count == 0:
                logger.info("No tags found. Seeding DBpedia ontology tags...")
                await seed_dbpedia_tags(session)
    except Exception as e:
        logger.warning(f"DBpedia seeding skipped: {e}")

    logger.info("Service ready!")
    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Semantic Tagging Service",
    description=(
        "A semantic tagging platform built on DBpedia Ontology. "
        "Uses bi-encoder (SentenceTransformers) + cross-encoder (reranker) pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tags.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "semantic-tagging"}


@app.get("/")
async def root():
    return {
        "service": "Semantic Tagging Service",
        "docs": "/docs",
        "health": "/health",
    }
