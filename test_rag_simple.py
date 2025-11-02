from qdrant_client import QdrantClient
import ollama

print("="*70)
print("RAG SYSTEM DEMONSTRATION - RETRIEVAL & EMBEDDING")
print("="*70)

query = "Explain angle of elevation"
print(f"\nQuery: {query}\n")

# Step 1: Embedding
print("[1/2] Generating 768-dimensional query embedding...")
try:
    emb = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
    print(f" Successfully generated embedding\n")
except Exception as e:
    print(f" Failed: {e}\n")
    exit(1)

# Step 2: Vector Search
print("[2/2] Searching Qdrant vector database...")
try:
    client = QdrantClient(host="localhost", port=6333)
    results = client.search(
        collection_name="trigonometry_chapter",
        query_vector=emb,
        limit=5
    )
    print(f"Retrieved {len(results)} relevant chunks\n")
except Exception as e:
    print(f"Failed: {e}\n")
    exit(1)

# Display results
print("="*70)
print("RETRIEVED CONTEXT (Semantic Search Results):")
print("="*70)

for i, result in enumerate(results, 1):
    page = result.payload['page'] + 1
    source = result.payload.get('source', 'text')
    similarity = result.score
    text = result.payload['text'][:180]
    
    print(f"\n[Result {i}] Page {page} ({source}) - Relevance: {similarity:.3f}")
    print(f"Text: {text}...\n")

print("="*70)
print("✓ RAG SYSTEM WORKING PERFECTLY!")
print("="*70)

