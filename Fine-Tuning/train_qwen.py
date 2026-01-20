# train_qwen.py

import json
import logging
import math
import os
import random
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# --- CONFIG 
# Get the absolute path of the folder where THIS script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level (..) to find the 'data' folder
DATA_FILE = os.path.join(SCRIPT_DIR, "../data/dnd_training_pairs.jsonl")

# Same logic for output (optional, but good practice)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "../output/fine_tuned_qwen_dnd")

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


# Batch size: 

# BATCH_SIZE = 2      # Small batch for CPU
# EPOCHS = 1          # 1 Epoch is enough to test
# WARMUP_STEPS = 5
# device = "cpu"      # Force CPU

BATCH_SIZE = 8       # Lower this if you get "Out of Memory" errors
EPOCHS = 10          # We set high, but we will save the BEST checkpoint automatically
WARMUP_STEPS = 50

# Setup Logging
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def load_data(filepath):
    """Reads JSONL and returns training examples + validation data dictionary"""
    train_examples = []
    
    # Validation data structures for the Evaluator
    val_queries = {}  # {qid: "query text"}
    val_corpus = {}   # {doc_id: "doc text"}
    val_relevant_docs = {} # {qid: {doc_id}}

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # Shuffle ensures our validation set is random
        random.shuffle(lines)
        
        # Split: 90% Train, 10% Validation
        split_idx = int(len(lines) * 0.9)
        train_lines = lines[:split_idx]
        val_lines = lines[split_idx:]
        
        # 1. Process Training Data
        for line in train_lines:
            data = json.loads(line)
            # Format: InputExample(texts=[Query, Positive_Context])
            train_examples.append(InputExample(texts=[data['query'], data['pos']]))
            
        # 2. Process Validation Data (Specific format for Evaluator)
        for idx, line in enumerate(val_lines):
            data = json.loads(line)
            qid = str(idx)
            doc_id = str(idx) # In our synthetic data, 1 query matches 1 doc
            
            val_queries[qid] = data['query']
            val_corpus[doc_id] = data['pos']
            val_relevant_docs[qid] = {doc_id}

    return train_examples, val_queries, val_corpus, val_relevant_docs

def main():
    logger.info(f"Loading Model: {MODEL_NAME}")
    # Initialize Qwen (Trust Remote Code is mandatory)
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    
    logger.info("Loading and Splitting Data...")
    train_examples, val_queries, val_corpus, val_rels = load_data(DATA_FILE)
    
    logger.info(f"Training Samples: {len(train_examples)} | Validation Samples: {len(val_queries)}")
    
    # 1. DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    
    # 2. Loss Function (MultipleNegativesRankingLoss is best for RAG)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # 3. Evaluator (The Metric Tracker)
    # This simulates a "Mini RAG" every epoch to see how well it finds the right documents.
    evaluator = InformationRetrievalEvaluator(
        val_queries, 
        val_corpus, 
        val_rels,
        name='dnd_validation',
        show_progress_bar=True
    )
    
    # 4. Run Training
    logger.info("Starting Training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        output_path=OUTPUT_PATH,
        evaluator=evaluator,          # Validate every epoch
        evaluation_steps=0,           # 0 = Run evaluation at end of epoch
        save_best_model=True,         # Only save if score improves
        show_progress_bar=True
    )
    
    logger.info(f"✅ Training Complete. Best model saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()