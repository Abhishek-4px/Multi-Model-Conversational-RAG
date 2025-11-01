"""
Caching manager for LLM responses.
Supports both prompt caching and conversational caching.
"""

import hashlib
import json
from typing import Optional, Dict, Any, List
import diskcache as dc
from datetime import datetime


class PromptCache:
    """Manages caching of LLM responses."""
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cache
        """
        self.cache = dc.Cache(cache_dir)
    
    def _generate_key(self, prompt: str, model: str) -> str:
        """
        Generate cache key from prompt and model.
        
        Args:
            prompt: User prompt
            model: Model name
            
        Returns:
            Cache key
        """
        content = f"{prompt}_{model}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, prompt: str, model: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response.
        
        Args:
            prompt: User prompt
            model: Model name
            
        Returns:
            Cached response or None
        """
        key = self._generate_key(prompt, model)
        cached = self.cache.get(key)
        
        if cached:
            print("✓ Cache hit! Retrieving from cache...")
            return cached
        
        return None
    
    def set(self, prompt: str, model: str, response: Dict[str, Any]):
        """
        Cache response.
        
        Args:
            prompt: User prompt
            model: Model name
            response: Response to cache
        """
        key = self._generate_key(prompt, model)
        cached_data = {
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "model": model
        }
        self.cache.set(key, cached_data, expire=3600)  # 1 hour expiry
        print("✓ Response cached")
    
    def clear(self):
        """Clear all cached data."""
        self.cache.clear()
        print("✓ Cache cleared")


class ConversationalMemory:
    """Manages conversational memory for multi-turn conversations."""
    
    def __init__(self, memory_key: str = "chat_history"):
        """
        Initialize conversational memory.
        
        Args:
            memory_key: Key to store conversation history
        """
        self.memory_key = memory_key
        self.messages: List[Dict[str, str]] = []
    
    def add_user_message(self, message: str):
        """
        Add user message to history.
        
        Args:
            message: User message
        """
        self.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_ai_message(self, message: str):
        """
        Add AI message to history.
        
        Args:
            message: AI message
        """
        self.messages.append({
            "role": "assistant",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_history(self, last_k: int = None) -> List[Dict[str, str]]:
        """
        Get conversation history.
        
        Args:
            last_k: Return only last k messages
            
        Returns:
            List of messages
        """
        if last_k:
            return self.messages[-last_k:]
        return self.messages
    
    def get_formatted_history(self) -> str:
        """
        Get formatted conversation history.
        
        Returns:
            Formatted string of conversation
        """
        formatted = []
        for msg in self.messages:
            role = "User" if msg["role"] == "user" else "AI"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)
    
    def clear(self):
        """Clear conversation history."""
        self.messages = []
        print("✓ Conversation memory cleared")
