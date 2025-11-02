import ollama
import time

def pull_model(model_name):
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"{'='*60}")
    
    try:
        stream = ollama.pull(model_name, stream=True)
        for chunk in stream:
            status = chunk.get('status', '')
            if status:
                print(f"\r{status}", end='', flush=True)
        
        print(f"\n{model_name} downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"\nError downloading {model_name}: {e}")
        return False

def list_models():
    try:
        result = ollama.list()
        print("\n" + "="*60)
        print("Available Models:")
        print("="*60)
        
        if not result.get('models'):
            print("No models installed yet")
            return []
        
        models = []
        for model in result['models']:
            name = model['name']
            size_gb = model['size'] / (1024**3)
            models.append(name)
            print(f"{name:30} ({size_gb:.2f} GB)")
        
        return models
        
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

def main():
    print("="*60)
    print("OLLAMA MODEL DOWNLOADER")
    print("="*60)

    print("\nChecking existing models")
    existing_models = list_models()
    
    required_models = ['nomic-embed-text', 'llama3']
    
    for model in required_models:
        if model in existing_models:
            print(f"\n✓ {model} already installed, skipping")
        else:
            success = pull_model(model)
            if not success:
                print(f"\n⚠ Failed to download {model}")
                # Ask if user wants to continue
                continue
    
    print("\n" + "="*60)
    print("FINAL STATUS")
    print("="*60)
    list_models()
    
    print("\nSetup complete")

if __name__ == "__main__":
    main()
