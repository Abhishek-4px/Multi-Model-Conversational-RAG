import ollama

print("Pulling phi model (smallest, ~2.7 GB)...")
print("This is the fastest and will definitely fit!")
print("-" * 60)

try:
    stream = ollama.pull('phi', stream=True)
    for chunk in stream:
        status = chunk.get('status', '')
        if status:
            print(f"\r{status}", end='', flush=True)
    
    print("\n✓ phi downloaded successfully!")
    
    # Verify
    print("\nAvailable models:")
    models = ollama.list()
    for model in models['models']:
        print(f"  ✓ {model['name']}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
