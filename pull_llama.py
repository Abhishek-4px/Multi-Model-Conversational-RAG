"""
Pull Ollama models using Python API
"""
import ollama
import time

def pull_model(model_name):
    """Pull a model with progress indication"""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Pull the model
        stream = ollama.pull(model_name, stream=True)
        
        # Show progress
        for chunk in stream:
            status = chunk.get('status', '')
            if status:
                print(f"\r{status}", end='', flush=True)
        
        print(f"\n✓ {model_name} downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading {model_name}: {e}")
        return False

def list_models():
    """List all available models"""
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
            print(f"✓ {name:30} ({size_gb:.2f} GB)")
        
        return models
        
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

def main():
    print("="*60)
    print("OLLAMA MODEL DOWNLOADER")
    print("="*60)
    
    # Check current models
    print("\nChecking existing models...")
    existing_models = list_models()
    
    # Models we need
    required_models = ['nomic-embed-text', 'llama3']
    
    # Download missing models
    for model in required_models:
        if model in existing_models:
            print(f"\n✓ {model} already installed, skipping...")
        else:
            success = pull_model(model)
            if not success:
                print(f"\n⚠ Failed to download {model}")
                # Ask if user wants to continue
                continue
    
    # Show final status
    print("\n" + "="*60)
    print("FINAL STATUS")
    print("="*60)
    list_models()
    
    print("\n✓ Setup complete! You can now run: python setup_pipeline.py")

if __name__ == "__main__":
    main()
