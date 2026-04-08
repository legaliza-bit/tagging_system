import logging
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://tagging:tagging@postgres:5432/tagging"
    DATABASE_URL_SYNC: str = "postgresql://tagging:tagging@postgres:5432/tagging"

    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_TAG_COLLECTION: str = "tags"
    QDRANT_DOC_COLLECTION: str = "documents"

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    TAG_DEDUP_THRESHOLD: float = 0.80
    TAG_ASSIGNMENT_THRESHOLD: float = 0.65
    TAG_SCORE_MARGIN: float = 0.10   # drop tags scoring more than this below the best
    TAG_FREQ_ALPHA: float = 0.30     # frequency penalty weight: score -= alpha * (count/total)
    TOP_K_CANDIDATES: int = 10
    TOP_K_TAGS: int = 5

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    BACKEND_URL: str = "http://backend:8000"

    DBPEDIA_CLASSES: list[str] = [
        "Company", "EducationalInstitution", "Artist", "Athlete",
        "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
        "Village", "Animal", "Plant", "Album", "Film", "WrittenWork",
    ]
    DBPEDIA_DESCRIPTIONS: dict[str, str] = {
        "Company":                "Business organizations, corporations, and commercial enterprises",
        "EducationalInstitution": "Schools, universities, colleges and other educational entities",
        "Artist":                 "Painters, sculptors, musicians, and other creative artists",
        "Athlete":                "Sports players and professional athletes",
        "OfficeHolder":           "Politicians, government officials, and public office holders",
        "MeanOfTransportation":   "Vehicles, aircraft, ships, trains and transportation modes",
        "Building":               "Structures, architecture, landmarks and built infrastructure",
        "NaturalPlace":           "Mountains, rivers, lakes, geographical and natural features",
        "Village":                "Small settlements, towns, hamlets and communities",
        "Animal":                 "Fauna, wildlife, mammals, birds, fish and other animals",
        "Plant":                  "Flora, vegetation, trees, flowers and botanical entities",
        "Album":                  "Music albums, records, and audio collections",
        "Film":                   "Movies, cinema, motion pictures and films",
        "WrittenWork":            "Books, novels, literature and written publications",
    }
    RERANKER_BASE_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    FINETUNED_MODEL_DIR: str = "/models/cross_encoder_dbpedia"


settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)