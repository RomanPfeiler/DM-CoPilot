==============================================================================
PROJECT: Dungeon Master Co-Pilot
==============================================================================

1. PROJECT SUMMARY
------------------------------------------------------------------------------

The Dungeon Master Co-Pilot is an AI assistant designed for Tabletop Roleplaying
Games (TTRPGs). Unlike standard chatbots, it "listens" to the game transcript 
in real-time. When it detects that the Dungeon Master (DM) or players are discussing 
rules or campaign lore, it proactively retrieves the relevant information from 
PDF rulebooks and campaign manuals, and presents it to the DM.


2. DATASET INFORMATION
------------------------------------------------------------------------------

The system relies on three distinct categories of data:

A. Knowledge Base
   * Rulebook: "D&D 5e Systems Reference Document" (PDF). Contains explicit game 
     mechanics (combat, spells, classes).
   * Campaign: Custom setting and lore document (PDF). Contains world-specific 
     facts (NPCs, locations, history).

B. Training Data
   * Format: JSONL containing synthetic (Query, Positive_Context) pairs.
   * Purpose: Used to fine-tune the embedding model via Contrastive Loss. 

C. Input Data
   * Source: Live-play session recordings converted to text.
   * Processing: The raw text is treated as a continuous stream, chunked into 
     70-word sliding windows to simulate real-time conversation flow.


3. ARCHITECTURE & APPROACH
------------------------------------------------------------------------------

A. The "Active Loop" (RAG Pipeline)
   Unlike standard RAG (which waits for a user query), this system runs a 
   continuous loop:
   1. Ingestion: PDFs are loaded (PyPDFLoader), split into 1500-char chunks 
      with overlap, and indexed in a local ChromaDB.
   2. Triggering: As transcript text arrives, it is vectorized.
   3. Thresholding: The system calculates Cosine Similarity between the live 
      transcript and the Knowledge Base. 
      - If Similarity Score < Threshold (0.75): Trigger Retrieval.
      - Else: Ignore (classified as "chit-chat").

B. Fine-Tuned Embeddings
   * Fine-tuned "Qwen/Qwen3-Embedding-0.6B" using SentenceTransformers 
     and MultipleNegativesRankingLoss.

C. Hybrid LLM Deployment
   * Cloud Mode: Google Gemini 2.0 Flash
   * Local Mode: Llama-3.2-1B via Ollama


4. RESULTS & ITERATION HISTORY
------------------------------------------------------------------------------

The project evolved through 6 major versions.
* Baseline: 
  - Setup: all-MiniLM-L6-v2 embeddings, small chunks (100 chars) with Google Gemini API call for answer.

* Optimization Phase:
  - Chunking Strategy: Increased document chunks to 1500 chars. 
  - Fine-Tuning: The custom Qwen model fine-tuned on rule book query chunks and synthentic positive context.
  - Use local LLM: Switching to local Llama-3.2-1B 


5. DISCUSSION
------------------------------------------------------------------------------


7. LIMITATIONS
------------------------------------------------------------------------------


8. QUICK START INSTRUCTIONS
------------------------------------------------------------------------------

A. Environment Setup
   1. Install Python 3.10+
   2. Create a virtual environment:
   3. Install dependencies:
      $ pip install -r requirements.txt

B. Configuration
   1. Create a .env file in the root directory.
   2. Add your API keys: GOOGLE_API_KEY=your_key_here
   3. Review src/config.py to toggle between "ollama" and "gemini".

C. Model Setup (Local)
   1. Install Ollama
   2. Pull the specific model used in V06:
      $ ollama pull llama3.2:1b

D. Directory Structure
   Ensure your data is placed correctly:
   /data
     /raw
       - dnd_rules.pdf
       - campaign.pdf
     /transcripts
       - live-play-transcript.txt

E. Running the System
   Step 1: Ingest Data
   Open "notebooks/01_setup_and_ingest.ipynb". Run all cells to parse PDFs, 
   load the fine-tuned embedding model, and build the ChromaDB vector store.

   Step 2: Run Simulation
   Open "notebooks/02_active_loop_simulation.ipynb". Run all cells to simulate 
   the game session. This will output a CSV file showing exactly when the AI 
   triggered and what it answered.
   
   Step 3: Fine-Tuning
   If you wish to retrain the embedding model, run "Fine-Tuning/train_qwen.py".
