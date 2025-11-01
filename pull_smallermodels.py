import ollama

# Try pulling the absolute smallest model that works
models_to_try = [
    'neural-chat:latest',  # ~4.1 GB (should fit exactly)
    'dolphin-mixtral:latest',  # ~8.9 GB (might not fit)
]

for model in models_to_try:
    print(f"\nTrying to pull {model}...")
    try:
        stream = ollama.pull(model, stream=True)
        for chunk in stream:
            status = chunk.get('status', '')
            if status:
                print(f"\r{status}", end='', flush=True)
        print(f"\n✓ {model} downloaded successfully!")
        break
    except Exception as e:
        print(f"\n✗ {model} failed: {str(e)[:100]}")
        continue

# Show available models
print("\n" + "="*60)
print("Available models:")
models = ollama.list()
for model in models['models']:
    print(f"  ✓ {model['name']}")
