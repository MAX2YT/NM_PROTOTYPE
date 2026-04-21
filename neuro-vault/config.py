"""
config.py — Central configuration for Neuro-Vault Clinical AI.

All paths, model names, thresholds, and constants are defined here.
No values should be hardcoded anywhere else in the project.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    """Central configuration class for Neuro-Vault.

    All system-wide constants, paths, and hyperparameters are
    defined here. Reference via ``from config import Config``.
    """

    # ------------------------------------------------------------------ #
    #  Base paths
    # ------------------------------------------------------------------ #
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    VECTOR_STORE_DIR: Path = DATA_DIR / "vector_store"
    EVAL_DIR: Path = DATA_DIR / "eval"
    AUDIT_DIR: Path = DATA_DIR / "audit"
    MODELS_DIR: Path = BASE_DIR / "models"

    # ------------------------------------------------------------------ #
    #  Dataset paths
    # ------------------------------------------------------------------ #
    MTSAMPLES_PATH: Path = RAW_DIR / "mtsamples.csv"
    MEDQUAD_DIR: Path = RAW_DIR / "MedQuAD"
    PUBMED_PATH: Path = RAW_DIR / "pubmed_abstracts.json"
    MEDQUAD_TEST_PATH: Path = EVAL_DIR / "medquad_test.json"

    # ------------------------------------------------------------------ #
    #  FAISS / Vector store
    # ------------------------------------------------------------------ #
    FAISS_INDEX_PATH: Path = VECTOR_STORE_DIR / "index.faiss"
    METADATA_PATH: Path = VECTOR_STORE_DIR / "metadata.json"

    # ------------------------------------------------------------------ #
    #  Models
    # ------------------------------------------------------------------ #
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "emilyalsentzer/Bio_ClinicalBERT"
    )
    EMBEDDING_MODEL_DIR: Path = MODELS_DIR / "bio_clinical_bert"
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    EMBEDDING_DIM: int = 768  # BioClinicalBERT output dimensionality

    # ------------------------------------------------------------------ #
    #  Ollama LLM
    # ------------------------------------------------------------------ #
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "phi3:mini")
    OLLAMA_TIMEOUT: int = 300  # seconds — phi3:mini on CPU can be slow
    OLLAMA_TEMPERATURE: float = 0.1

    # ------------------------------------------------------------------ #
    #  Chunking
    # ------------------------------------------------------------------ #
    CHUNK_SIZE: int = 400          # tokens (approximate word count)
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_LENGTH: int = 50     # discard chunks shorter than this

    # ------------------------------------------------------------------ #
    #  Retrieval
    # ------------------------------------------------------------------ #
    DENSE_TOP_K: int = 10
    SPARSE_TOP_K: int = 10
    FINAL_TOP_K: int = 5
    RRF_K: int = 60

    # Dataset authority weights for RRF score multiplication
    DATASET_WEIGHTS: dict = {
        "mtsamples": 1.0,
        "medquad": 1.2,
        "pubmed": 1.5,
    }

    # ------------------------------------------------------------------ #
    #  Abstention thresholds
    # ------------------------------------------------------------------ #
    RERANK_THRESHOLD: float = 0.45
    COSINE_THRESHOLD: float = 0.60
    ENTITY_COVERAGE_THRESHOLD: float = 0.50

    # ------------------------------------------------------------------ #
    #  Security / Encryption
    # ------------------------------------------------------------------ #
    ENCRYPTION_KEY: str = os.getenv(
        "NEURO_VAULT_KEY", "neurovault-demo-key-change-in-prod"
    )
    AUDIT_DB_PATH: Path = AUDIT_DIR / "audit.db"

    # ------------------------------------------------------------------ #
    #  PubMed E-utilities
    # ------------------------------------------------------------------ #
    PUBMED_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    PUBMED_RETMAX: int = 50
    PUBMED_BATCH_SIZE: int = 20
    PUBMED_SLEEP: float = 0.34   # seconds between requests (≤3 req/s)
    PUBMED_MAX_RETRIES: int = 3

    PUBMED_QUERIES: list = [
        "Type 2 Diabetes Mellitus treatment guidelines India",
        "Tuberculosis management protocol India",
        "Hypertension treatment South Asia",
        "Dengue fever clinical management",
        "Malaria treatment guidelines India",
        "Snake bite management India",
        "Typhoid fever antibiotic treatment",
        "COPD management primary care",
        "Heart failure treatment guidelines",
        "Stroke management acute care",
        "Sepsis treatment protocol ICU",
        "Acute kidney injury management",
        "Anemia treatment iron deficiency India",
        "Pediatric pneumonia management India",
        "Maternal mortality prevention India",
        "DPDP Act health data India",
        "Electronic health records India hospital",
        "Drug interaction clinical pharmacology",
        "Antimicrobial resistance India",
        "COVID-19 clinical management India",
    ]

    # ------------------------------------------------------------------ #
    #  Ingestion batch sizes
    # ------------------------------------------------------------------ #
    INGEST_BATCH_SIZE: int = 100
    EMBED_BATCH_SIZE: int = 32

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #
    MEDQUAD_TEST_RATIO: float = 0.10   # stratified per source folder

    # ------------------------------------------------------------------ #
    #  Helper utilities
    # ------------------------------------------------------------------ #
    @classmethod
    def ensure_dirs(cls) -> None:
        """Create all required directories if they do not exist."""
        for path in [
            cls.DATA_DIR,
            cls.RAW_DIR,
            cls.VECTOR_STORE_DIR,
            cls.EVAL_DIR,
            cls.AUDIT_DIR,
            cls.MODELS_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def index_exists(cls) -> bool:
        """Return True if the FAISS index has already been built."""
        return cls.FAISS_INDEX_PATH.exists() and cls.METADATA_PATH.exists()

    @classmethod
    def dataset_status(cls) -> dict:
        """Return availability status of each dataset.

        Returns:
            dict: Keys are dataset names; values are bool indicating
                  whether the source file/directory is present.
        """
        return {
            "mtsamples": cls.MTSAMPLES_PATH.exists(),
            "medquad": cls.MEDQUAD_DIR.exists()
            and any(cls.MEDQUAD_DIR.rglob("*.xml")),
            "pubmed": cls.PUBMED_PATH.exists(),
        }
