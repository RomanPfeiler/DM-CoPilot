# src/llm_factory.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from config import LLM_PROVIDER, GOOGLE_MODEL_NAME, OLLAMA_MODEL_NAME

def get_llm():
    """
    Returns the configured LLM instance (Gemini or Ollama) based on config.py.
    """
    print(f"🏭 Factory Initializing LLM: {LLM_PROVIDER.upper()}")
    
    if LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(
            model=GOOGLE_MODEL_NAME,
            temperature=0
        )
    
    elif LLM_PROVIDER == "ollama":
        # keep_alive="-1" forces the model to stay in RAM (speed optimization)
        return ChatOllama(
            model=OLLAMA_MODEL_NAME,
            temperature=0,
            keep_alive=-1
        )
    
    else:
        raise ValueError(f"Unknown LLM_PROVIDER in config: {LLM_PROVIDER}")