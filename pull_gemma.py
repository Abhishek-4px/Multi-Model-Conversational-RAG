import ollama

print("Pulling gemma3:4b model (2.6 GB)...")
print("This may take 2-3 minutes...")
print("-" * 60)

try:
    stream = ollama.pull('gemma3:4b', stream=True)
    for chunk in stream:
        status = chunk.get('status', '')
        if status:
            print(f"\r{status}", end='', flush=True)
    
    print("\n✓ gemma3:4b downloaded successfully!")
    
    # Verify both models
    print("\nAvailable models:")
    models = ollama.list()
    for model in models['models']:
        print(f"  ✓ {model['name']}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
