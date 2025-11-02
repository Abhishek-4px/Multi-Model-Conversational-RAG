import ollama
models_to_try = [
    'neural-chat:latest', 
    'dolphin-mixtral:latest', 
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

print("\n" + "="*60)
print("Available models:")
models = ollama.list()
for model in models['models']:
    print(f" {model['name']}")
