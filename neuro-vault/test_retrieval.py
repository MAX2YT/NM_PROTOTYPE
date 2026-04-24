from config import Config
from ingest.embedder import Embedder
from retrieval.retriever import HybridRetriever

# Load pipeline
embedder = Embedder(
    model_name=Config.EMBEDDING_MODEL,
    model_dir=Config.EMBEDDING_MODEL_DIR,
    index_path=Config.FAISS_INDEX_PATH,
    metadata_path=Config.METADATA_PATH,
)
embedder.load()
retriever = HybridRetriever(embedder)

query = "What are the treatment guidelines for Type 2 Diabetes Mellitus?"
retriever.final_top_k = 3
chunks = retriever.retrieve(query)

print(f"Query: {query}")
print(f"Number of chunks retrieved: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(f"Dataset: {chunk.get('dataset')}")
    print(f"Text Snippet: {chunk.get('text')[:200]}...")
    print(f"Contains 'diabetes mellitus': {'diabetes mellitus' in chunk.get('text').lower()}")
