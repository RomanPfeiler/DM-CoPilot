# src/config.py

import os
from dotenv import load_dotenv

# ---------------------------------------------------------
# 1. SETUP & PATHS
# ---------------------------------------------------------
# Load environment variables (API keys) from the .env file in the root
load_dotenv()

# Project Root (calculated from src/config.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths for data
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")

# ---------------------------------------------------------
# 2. EMBEDDING MODEL SETTINGS (The "Retriever")
# ---------------------------------------------------------
# We point this to your local fine-tuned model folder.

# To switch back to the original:
# EMBEDDING_MODEL_PATH = "Qwen/Qwen3-Embedding-0.6B" 
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "Output", "fine_tuned_qwen_dnd")

# ---------------------------------------------------------
# 3. LLM SETTINGS (The "Chat Bot")
# ---------------------------------------------------------
# This controls which AI answers the user.

# TOGGLE THIS: Choose "gemini" or "ollama"
LLM_PROVIDER = "ollama"

# Option A: Google Gemini (Cloud)
GOOGLE_MODEL_NAME = "gemini-2.0-flash"

# Option B: Ollama (Local)
OLLAMA_MODEL_NAME = "llama3.2:1b"

# ---------------------------------------------------------
# Other models used:
# ---------------------------------------------------------
# We use a small local model
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# The off-the-shelve model:
# EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Model for the fine tuning data generation:
LLM_MODEL_NAME = "gemini-2.0-flash"

# ---------------------------------------------------------
# 4. TUNING PARAMETERS
# ---------------------------------------------------------
# Transcript Processing
CHUNK_SIZE = 70       # Words per chunk
OVERLAP = 20          # Overlap to keep context

# Retrieval Logic
TOP_K = 3             # How many chunks to fetch per source (Rules/Campaign)
TRIGGER_THRESHOLD = 0.75 # Distance score 

# Rate Limiting (0 for local)
API_DELAY_SECONDS = 0