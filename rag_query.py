import os
import sys
import argparse
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import ollama
import time
from typing import List, Dict, Any, Optional
from utils.cache_manager import PromptCache, ConversationalMemory

load_dotenv()

# Configs
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "trigonometry_chapter")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Init cache and memory
prompt_cache = PromptCache()
conversation_memory = ConversationalMemory()

# Uses ollama to generate the embeddings
def generate_embeddings(text: str, model: str = EMBEDDING_MODEL) -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# Takes user query , qdrant client  and results to retrieve for a quick dmeo I have done it to 5 , Gives you a list of relevent docs 
def retrieve_context(query: str, client: QdrantClient, top_k: int = 5) -> list:
    # Generating query embeddings
    query_embedding = generate_embeddings(query)
    
    # Searching in qdrant
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k
    )
    
    return results

# Summarization is not done using ollama but used groq we pass the list we got from retrieve_context , groq summarizes
def summarize_context(contexts: list) -> str:
    from groq import Groq
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not in .env file!")
    
    client = Groq(api_key=GROQ_API_KEY)
    
    # Combining contexts
    combined_text = "\n\n".join([
        f"Source {i+1} (Page {ctx.payload['page']+1}): {ctx.payload['text'][:300]}..." 
        for i, ctx in enumerate(contexts)
    ])
    
    # Summarization prompt created from prompt genie
    prompt = f"""You are analyzing content from a mathematics textbook chapter on trigonometry applications.

Retrieved Context:
{combined_text}

Provide a concise summary (2-3 sentences) highlighting the key mathematical concepts, formulas, or examples mentioned in the retrieved content:"""
    
    # GROQ generating summaries 
    message = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        temperature=0.7,
    )
    return message.choices[0].message.content


# This is the final answer generated , we take question , list of 5 top k relevent strings , returns the para
def generate_answer(question: str, contexts: List[str], use_memory: bool = False) -> str:
    from groq import Groq
    
    # Validate API key
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not in .env file!")

    client = Groq(api_key=GROQ_API_KEY)
 
    context_str = "\n\n".join([f"[Source {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])
    
    # Prompt is taken from www.promptgenie.com
    prompt = f"""You are an expert academic tutor specializing in mathematics, especially trigonometry and geometry. 
Answer the question based ONLY on the provided context. Be clear, concise, and educational.

Context:
{context_str}

Question: {question}

Answer:"""
    
    # Using groq not ollama
    print("Generating answer using Groq (llama-3.1-70b)...")
    message = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        temperature=0.7,
    )
    
    return message.choices[0].message.content

# formatting of source citations as req in assignment
def format_sources(contexts: list) -> str:

    sources = []
    for i, ctx in enumerate(contexts):
        page = ctx.payload['page'] + 1
        source_type = ctx.payload.get('source', 'text')
        text = ctx.payload['text']
        
        preview = text[:150].replace('\n', ' ') + "..." if len(text) > 150 else text.replace('\n', ' ')
        
        if source_type == 'image':
            image_file = ctx.payload.get('image_filename', 'unknown')
            sources.append(f"  [{i+1}] Page {page} (DIAGRAM: {image_file})")
            sources.append(f"      {preview}")
        else:
            sources.append(f"  [{i+1}] Page {page} (text)")
            sources.append(f"      {preview}")
    
    return "\n".join(sources)

# The main part is from this block, You pass a query , with 3 more args
def run_query(question: str, summarize: bool = False, use_cache: bool = True, 
              conversational: bool = False):
    print("\n" + "=" * 70)
    print(f"QUERY: {question}")
    print("=" * 70)
    
    start_time = time.time()
    
    cached_response = None
    if use_cache:
        cache_key = f"{question}_{summarize}_{conversational}"
        cached_response = prompt_cache.get(cache_key, "groq")
    
    if cached_response:
        elapsed = time.time() - start_time
        print(f"\n⚡ Retrieved from cache (Time: {elapsed:.2f}s)")
        print("\n" + "-" * 70)
        print("FINAL ANSWER:")
        print("-" * 70)
        print(cached_response["response"]["answer"])
        print("\n" + "-" * 70)
        print("SOURCES:")
        print("-" * 70)
        print(cached_response["response"]["sources"])
        return
    
    # Init the Qdrant client 
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    except Exception as e:
        print(f"\n Error connecting to Qdrant: {e}")
        print("Make sure Qdrant is running: docker-compose up -d")   # better for debugging , wasted 20 mins here :(
        return
    
    # First is retrieve the context
    print(f"\n[1/3] Retrieving relevant context from Qdrant...")
    contexts = retrieve_context(question, client, top_k=5)
    
    if not contexts:
        print("No relevant context found. Make sure setup_pipeline.py has been run.")
        return
    
    print(f"Retrieved {len(contexts)} relevant chunks")
    
    # Printing some retrieved stuff
    text_sources = sum(1 for c in contexts if c.payload.get('source') == 'text')
    image_sources = sum(1 for c in contexts if c.payload.get('source') == 'image')
    if image_sources > 0:
        print(f"  - Text chunks: {text_sources}")
        print(f"  - Diagram chunks: {image_sources}")
    
    # Summarization using groq (ollama can't be pulled due to less space)
    if summarize:
        print(f"\n[2/3] Generating summary of retrieved context using Groq...")
        try:
            summary = summarize_context(contexts)
            print("\n" + "-" * 70)
            print("RETRIEVED CONTEXT SUMMARY:")
            print("-" * 70)
            print(summary)
            print("-" * 70)
        except Exception as e:
            print(f"Summarization failed: {e}")
    else:
        print(f"\n[2/3] Skipping summarization...")
    
    # Generating ans now
    print(f"\n[3/3] Generating answer using Groq...")
    
    # Adding conversational memory 
    if conversational:
        conversation_memory.add_user_message(question)
    
    # Extracting text from contexts
    context_texts = [ctx.payload['text'] for ctx in contexts]
    
    try:
        answer = generate_answer(question, context_texts, use_memory=conversational)
    except Exception as e:
        print(f"\n Error generating answer: {e}")
        return
    
    if conversational:
        conversation_memory.add_ai_message(answer)
    
    sources = format_sources(contexts)
    
    elapsed_time = time.time() - start_time
    
    # To Cache responses
    if use_cache:
        response_data = {
            "answer": answer,
            "sources": sources,
            "time": elapsed_time
        }
        prompt_cache.set(f"{question}_{summarize}_{conversational}", "groq", response_data)

    print("\n" + "=" * 70)
    print(f"✓ QUERY COMPLETE (Time: {elapsed_time:.2f}s)")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("FINAL ANSWER:")
    print("-" * 70)
    print(answer)
    
    print("\n" + "-" * 70)
    print("SOURCES:")
    print("-" * 70)
    print(sources)
    print()

# this is the main entry point
def main():
    parser = argparse.ArgumentParser(
        description="RAG Query System for Trigonometry Chapter using Groq API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_query.py --question "Explain the steps involved in solving problems using angle of elevation"
  python rag_query.py --question "What does the diagram in Figure 9.4 show?" 
  python rag_query.py --question "What is angle of depression?" --summarize
  python rag_query.py --question "Who can use trigonometry?" --conversational
        """
    )
    parser.add_argument("--question", "-q", type=str, required=True, 
                       help="Question to ask about the trigonometry chapter")
    parser.add_argument("--summarize", "-s", action="store_true", 
                       help="Show context summarization before final answer")
    parser.add_argument("--no-cache", action="store_true", 
                       help="Disable caching (force new LLM generation)")
    parser.add_argument("--conversational", "-c", action="store_true", 
                       help="Enable conversational memory for follow-up questions")
    parser.add_argument("--clear-cache", action="store_true", 
                       help="Clear cache before running query")
    parser.add_argument("--clear-memory", action="store_true", 
                       help="Clear conversation memory")
    
    args = parser.parse_args()
    
    if args.clear_cache:
        prompt_cache.clear()
        print("Cache cleared")
    
    if args.clear_memory:
        conversation_memory.clear()
        print("Conversation memory cleared")
    
    run_query(
        question=args.question,
        summarize=args.summarize,
        use_cache=not args.no_cache,   # double negative for caching
        conversational=args.conversational
    )


if __name__ == "__main__":
    main()