import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama
from tqdm import tqdm

from utils.pdf_parser import MultimodalPDFParser
from utils.chunking import AcademicChunker

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "trigonometry_chapter")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
PDF_PATH = os.getenv("PDF_PATH", "jemh109.pdf")

# making text to embeds
def generate_embeddings(text: str, model: str = EMBEDDING_MODEL) -> list:
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        print(f"ollama pull {model}")
        return None

#
def setup_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    
    if collection_name in collection_names:
        print(f"Collection '{collection_name}' already exists. Deleting ")
        client.delete_collection(collection_name)
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"✓ Created collection '{collection_name}' with vector size {vector_size}")


def index_documents(chunks: list, client: QdrantClient, collection_name: str):
    """
    Index document chunks in Qdrant (OPTIMIZED).
    
    Args:
        chunks: List of document chunks
        client: Qdrant client
        collection_name: Name of collection
    """
    points = []
    batch_size = 10  # Process 10 at a time
    
    print(f"\nGenerating embeddings for {len(chunks)} chunks...")
    print("(Optimized batch processing - should take 2-3 minutes)\n")
    
    for idx, chunk in enumerate(tqdm(chunks, desc="Processing")):
        text = chunk["text"]
        metadata = chunk["metadata"]
        
        # Generate embedding
        embedding = generate_embeddings(text)
        
        if embedding:
            point = PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "text": text,
                    "page": metadata.get("page", -1),
                    "source": metadata.get("source", "unknown"),
                    "chunk_index": metadata.get("chunk_index", idx),
                    **metadata
                }
            )
            points.append(point)
        
        # Upload in batches of 10
        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            points = []
    
    # Upload remaining points
    if points:
        client.upsert(collection_name=collection_name, points=points)
    
    print(f"\n✓ Successfully indexed {len(chunks)} chunks")

        
        # Batch upload every 50 points for efficiency
    if len(points) >= 50:
            client.upsert(collection_name=collection_name, points=points)
            points = []
    
    # Upload remaining points
    if points:
        client.upsert(collection_name=collection_name, points=points)
    
    print(f"\n✓ Successfully indexed {len(chunks)} chunks")


def main():
    """Main setup pipeline."""
    print("=" * 70)
    print("RAG SYSTEM SETUP PIPELINE")
    print("Chapter 9: Some Applications of Trigonometry (jemh109.pdf)")
    print("=" * 70)
    
    # Step 1: Check PDF exists
    print(f"\n[1/5] Checking PDF file...")
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: PDF file not found at '{PDF_PATH}'")
        print("Please ensure jemh109.pdf is in the project directory.")
        return
    print(f"✓ PDF file found: {PDF_PATH}")
    
    # Step 2: Parse PDF
    print(f"\n[2/5] Parsing PDF (extracting text, images, and formulas)...")
    parser = MultimodalPDFParser(PDF_PATH)
    parser.open()
    pages = parser.parse_full_document()
    parser.close()
    
    total_images = sum(len(page["images"]) for page in pages)
    total_math_blocks = sum(page.get("math_block_count", 0) for page in pages)
    
    print(f"\n✓ Extracted content from {len(pages)} pages")
    print(f"✓ Found {total_images} images/diagrams (trigonometry figures)")
    print(f"✓ Detected {total_math_blocks} blocks with mathematical formulas")
    
    # Step 3: Chunk documents
    print(f"\n[3/5] Chunking documents with context preservation...")
    print(f"  - Chunk size: {CHUNK_SIZE}")
    print(f"  - Chunk overlap: {CHUNK_OVERLAP}")
    chunker = AcademicChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = chunker.chunk_document(pages)
    
    text_chunks = [c for c in chunks if c["metadata"]["source"] == "text"]
    image_chunks = [c for c in chunks if c["metadata"]["source"] == "image"]
    
    print(f"✓ Created {len(chunks)} chunks total")
    print(f"  - Text chunks: {len(text_chunks)}")
    print(f"  - Diagram/image chunks: {len(image_chunks)}")
    
    # Step 4: Setup Qdrant
    print(f"\n[4/5] Setting up Qdrant vector database...")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print(f"✓ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        print("Make sure Qdrant is running:")
        print("  docker-compose up -d")
        return
    
    # Get vector size from embedding model
    print(f"✓ Testing embedding model: {EMBEDDING_MODEL}")
    test_embedding = generate_embeddings("test")
    if not test_embedding:
        return
    
    vector_size = len(test_embedding)
    print(f"✓ Embedding dimension: {vector_size}")
    
    setup_qdrant_collection(client, COLLECTION_NAME, vector_size)
    
    # Step 5: Index documents
    print(f"\n[5/5] Indexing documents in Qdrant...")
    index_documents(chunks, client, COLLECTION_NAME)
    
    print("\n" + "=" * 70)
    print("✓ SETUP COMPLETE!")
    print("=" * 70)
    print(f"\nQdrant collection '{COLLECTION_NAME}' created.")
    print(f"Total documents indexed: {len(chunks)}")
    print(f"\nYou can now run queries using:")
    print(f"  python rag_query.py --question \"Your question here\"")
    print("\nExample queries:")
    print("  python rag_query.py --question \"Explain angle of elevation\"")
    print("  python rag_query.py --question \"What does Figure 9.4 show?\"")


if __name__ == "__main__":
    main()
