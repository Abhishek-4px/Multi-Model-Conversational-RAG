import hashlib
import json
from typing import Optional, Dict, Any, List
import diskcache as dc
from datetime import datetime

class PromptCache:   
    def __init__(self, cache_dir: str = "./cache"):
        self.cache = dc.Cache(cache_dir)
    
    def _generate_key(self, prompt: str, model: str) -> str:
        content = f"{prompt}_{model}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, prompt: str, model: str) -> Optional[Dict[str, Any]]:
        key = self._generate_key(prompt, model)
        cached = self.cache.get(key)
        
        if cached:
            print("Cache hit! Retrieving from cache")
            return cached
        
        return None
    
    def set(self, prompt: str, model: str, response: Dict[str, Any]):
        key = self._generate_key(prompt, model)
        cached_data = {
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "model": model
        }
        self.cache.set(key, cached_data, expire=3600)  # 1 hour expiry
        print("Response cached")
    
    def clear(self):
        self.cache.clear()
        print("Cache cleared")


class ConversationalMemory:  
    def __init__(self, memory_key: str = "chat_history"):
        self.memory_key = memory_key
        self.messages: List[Dict[str, str]] = []
    
    def add_user_message(self, message: str):
        self.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_ai_message(self, message: str):
        self.messages.append({
            "role": "assistant",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_history(self, last_k: int = None) -> List[Dict[str, str]]:
        if last_k:
            return self.messages[-last_k:]
        return self.messages
    
    def get_formatted_history(self) -> str:
        formatted = []
        for msg in self.messages:
            role = "User" if msg["role"] == "user" else "AI"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)
    
    def clear(self):
        self.messages = []
        print("Conversation memory cleared")
