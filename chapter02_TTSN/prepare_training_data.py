"""
Phase 1: Training Data Preparation for Two-Tower Model

This script prepares training data for the two-tower architecture:
1. Creates positive pairs from ESCI labels (E, S, C) with weights
2. Samples random negatives
3. Generates hard negatives using Chapter 1 FAISS system
4. Creates train/val/test splits
5. Saves training data in parquet format
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings
from collections import defaultdict
import random

warnings.filterwarnings('ignore')
tqdm.pandas()

# Set up paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR
CHAPTER1_DIR = ROOT_DIR / "chapter01_basic_retrieval_and_query_expansion_reranking"
CHAPTER1_DATA_DIR = CHAPTER1_DIR / "data"
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ESCI label weights
ESCI_WEIGHTS = {
    'E': 1.0,   # Exact (strong positive)
    'S': 0.9,   # Substitute (positive)
    'C': 0.6,   # Complement (weak positive)
    'I': 0.0    # Irrelevant (negative)
}


def load_data():
    """Load examples, products, and Chapter 1 indices"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # Load examples dataset
    print("\n1. Loading examples dataset...")
    examples_df = pd.read_parquet(DATA_DIR / "shopping_queries_dataset_examples.parquet")
    print(f"   Loaded {len(examples_df):,} examples")
    print(f"   Columns: {examples_df.columns.tolist()}")
    print(f"   ESCI distribution:")
    print(examples_df['esci_label'].value_counts())
    print(f"   Split distribution:")
    print(examples_df['split'].value_counts())
    
    # Load products dataset
    # Note: products_clean.parquet already contains 'product_text' field which combines:
    # - product_title
    # - product_description  
    # - product_bullet_point
    # - product_brand (with "Brand: " prefix)
    # - product_color (with "Color: " prefix)
    # This was computed in Chapter 1's build_index.py using create_product_text()
    print("\n2. Loading products dataset...")
    products_df = pd.read_parquet(CHAPTER1_DATA_DIR / "products_clean.parquet")
    print(f"   Loaded {len(products_df):,} products")
    
    # Verify product_text column exists (should be pre-computed from Chapter 1)
    if 'product_text' not in products_df.columns:
        raise ValueError("products_clean.parquet must contain 'product_text' column. "
                        "Please run Chapter 1's build_index.py first.")
    print(f"   Product text field available: {products_df['product_text'].notna().sum():,} products")
    
    # Filter examples to only include products that exist
    print("\n3. Filtering examples to existing products...")
    products_in_catalog = set(products_df['product_id'].unique())
    examples_df = examples_df[examples_df['product_id'].isin(products_in_catalog)].copy()
    print(f"   Filtered to {len(examples_df):,} examples with valid products")
    
    # Load Chapter 1 FAISS indices for hard negative mining
    print("\n4. Loading Chapter 1 FAISS indices...")
    with open(CHAPTER1_DATA_DIR / "metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    model_name = metadata['model_name']
    embedding_dim = metadata['embedding_dim']
    
    print(f"   Model: {model_name}")
    print(f"   Embedding dimension: {embedding_dim}")
    
    # Load embedding model
    embedding_model = SentenceTransformer(model_name)
    
    # Load FAISS index (use HNSW for faster retrieval)
    hnsw_index = faiss.read_index(str(CHAPTER1_DATA_DIR / "faiss_index_hnsw.bin"))
    hnsw_index.hnsw.efSearch = 200  # Set for better recall
    print(f"   HNSW index loaded: {hnsw_index.ntotal:,} vectors")
    
    # Load product IDs mapping
    with open(CHAPTER1_DATA_DIR / "product_ids.pkl", 'rb') as f:
        product_ids = pickle.load(f)
    print(f"   Product IDs loaded: {len(product_ids):,}")
    
    # Create product_id to index mapping
    product_id_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
    
    # Create product_id to text mapping for faster lookup
    product_id_to_text = dict(zip(products_df['product_id'], products_df['product_text']))
    
    return (examples_df, products_df, embedding_model, hnsw_index, 
            product_ids, product_id_to_idx, product_id_to_text, embedding_dim)


def create_positive_pairs(examples_df, products_df):
    """Create positive pairs from ESCI labels with weights"""
    print("\n" + "=" * 80)
    print("CREATING POSITIVE PAIRS")
    print("=" * 80)
    
    # Filter to positive labels (E, S, C)
    positive_labels = ['E', 'S', 'C']
    positive_examples = examples_df[examples_df['esci_label'].isin(positive_labels)].copy()
    
    print(f"\nPositive examples: {len(positive_examples):,}")
    print(f"Label distribution:")
    print(positive_examples['esci_label'].value_counts())
    
    # Create positive pairs with weights and stage markers
    positive_pairs = []
    
    for _, row in tqdm(positive_examples.iterrows(), total=len(positive_examples), desc="Creating positive pairs"):
        esci_label = row['esci_label']
        weight = ESCI_WEIGHTS[esci_label]
        
        # Determine which stages this example should be used in
        # Stage 1: E + S only (no C)
        # Stage 2: E + S only (no C)
        # Stage 3: E + S + C (all positives)
        use_in_stage_1 = esci_label in ['E', 'S']
        use_in_stage_2 = esci_label in ['E', 'S']
        use_in_stage_3 = True  # All positives (E, S, C) in Stage 3
        
        positive_pairs.append({
            'query': row['query'],
            'query_id': row['query_id'],
            'product_id': row['product_id'],
            'esci_label': esci_label,
            'label': 1.0,
            'weight': weight,
            'neg_type': 'positive',
            'split': row['split'],
            'use_in_stage_1': use_in_stage_1,
            'use_in_stage_2': use_in_stage_2,
            'use_in_stage_3': use_in_stage_3
        })
    
    positive_df = pd.DataFrame(positive_pairs)
    print(f"\nCreated {len(positive_df):,} positive pairs")
    print(f"Weight distribution:")
    print(positive_df['weight'].value_counts().sort_index(ascending=False))
    
    # Print stage distribution
    print(f"\nStage distribution for positives:")
    print(f"  Stage 1 (E+S): {(positive_df['use_in_stage_1']).sum():,}")
    print(f"  Stage 2 (E+S): {(positive_df['use_in_stage_2']).sum():,}")
    print(f"  Stage 3 (E+S+C): {(positive_df['use_in_stage_3']).sum():,}")
    print(f"  E labels: {(positive_df['esci_label'] == 'E').sum():,}")
    print(f"  S labels: {(positive_df['esci_label'] == 'S').sum():,}")
    print(f"  C labels: {(positive_df['esci_label'] == 'C').sum():,}")
    
    return positive_df


def sample_random_negatives(examples_df, products_df, positive_df, 
                           neg_ratio=3, seed=42):
    """Sample random negative pairs"""
    print("\n" + "=" * 80)
    print("SAMPLING RANDOM NEGATIVES")
    print("=" * 80)
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Get all product IDs
    all_product_ids = set(products_df['product_id'].unique())
    
    # Group positives by query to get relevant products per query
    query_to_relevant = defaultdict(set)
    for _, row in positive_df.iterrows():
        query_to_relevant[row['query']].add(row['product_id'])
    
    print(f"\nUnique queries with positives: {len(query_to_relevant):,}")
    
    # Sample random negatives
    random_negatives = []
    n_positives = len(positive_df)
    n_negatives_target = n_positives * neg_ratio
    
    print(f"\nTarget negative:positive ratio: {neg_ratio}:1")
    print(f"Target negatives: {n_negatives_target:,}")
    
    # Sample negatives per query
    queries_with_positives = positive_df['query'].unique()
    
    for query in tqdm(queries_with_positives, desc="Sampling random negatives"):
        relevant_products = query_to_relevant[query]
        candidate_negatives = list(all_product_ids - relevant_products)
        
        # Sample negatives for this query
        n_negatives_per_query = len(relevant_products) * neg_ratio
        n_negatives_per_query = min(n_negatives_per_query, len(candidate_negatives))
        
        if n_negatives_per_query > 0:
            sampled_negatives = random.sample(candidate_negatives, n_negatives_per_query)
            
            # Get split from positive examples for this query
            query_positives = positive_df[positive_df['query'] == query]
            split = query_positives.iloc[0]['split'] if len(query_positives) > 0 else 'train'
            
            for product_id in sampled_negatives:
                # Random negatives are used in all stages
                random_negatives.append({
                    'query': query,
                    'query_id': query_positives.iloc[0]['query_id'] if len(query_positives) > 0 else None,
                    'product_id': product_id,
                    'esci_label': 'I',
                    'label': 0.0,
                    'weight': 0.0,
                    'neg_type': 'random',
                    'split': split,
                    'use_in_stage_1': True,   # Stage 1: Random negatives only
                    'use_in_stage_2': True,  # Stage 2: Random + Hard negatives
                    'use_in_stage_3': True   # Stage 3: Random + Hard negatives
                })
    
    random_neg_df = pd.DataFrame(random_negatives)
    print(f"\nCreated {len(random_neg_df):,} random negative pairs")
    print(f"Actual negative:positive ratio: {len(random_neg_df)/n_positives:.2f}:1")
    
    # Print stage distribution
    print(f"\nStage distribution for random negatives:")
    print(f"  Stage 1: {(random_neg_df['use_in_stage_1']).sum():,}")
    print(f"  Stage 2: {(random_neg_df['use_in_stage_2']).sum():,}")
    print(f"  Stage 3: {(random_neg_df['use_in_stage_3']).sum():,}")
    
    return random_neg_df


def generate_hard_negatives(examples_df, positive_df, embedding_model, 
                           hnsw_index, product_ids, product_id_to_idx,
                           product_id_to_text, k=100, top_k_hard=20, seed=42):
    """Generate hard negatives using Chapter 1 FAISS system"""
    print("\n" + "=" * 80)
    print("GENERATING HARD NEGATIVES")
    print("=" * 80)
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Group positives by query
    query_to_relevant = defaultdict(set)
    query_to_query_id = {}
    query_to_split = {}
    
    for _, row in positive_df.iterrows():
        query = row['query']
        query_to_relevant[query].add(row['product_id'])
        query_to_query_id[query] = row['query_id']
        query_to_split[query] = row['split']
    
    print(f"\nGenerating hard negatives for {len(query_to_relevant):,} queries")
    print(f"Retrieving top-{k} candidates, keeping top-{top_k_hard} as hard negatives")
    
    hard_negatives = []
    
    # Process queries in batches for efficiency
    batch_size = 32
    queries_list = list(query_to_relevant.keys())
    
    for i in tqdm(range(0, len(queries_list), batch_size), desc="Generating hard negatives"):
        batch_queries = queries_list[i:i+batch_size]
        
        # Embed queries
        query_embeddings = embedding_model.encode(
            batch_queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype('float32')
        
        # Search in FAISS
        distances, indices = hnsw_index.search(query_embeddings, k)
        
        # Process results for each query
        for query_idx, query in enumerate(batch_queries):
            relevant_products = query_to_relevant[query]
            retrieved_indices = indices[query_idx]
            retrieved_distances = distances[query_idx]
            
            # Filter out positives and get hard negatives
            hard_neg_candidates = []
            
            for idx, dist in zip(retrieved_indices, retrieved_distances):
                if idx < len(product_ids):
                    product_id = product_ids[idx]
                    
                    # Skip if it's a positive
                    if product_id in relevant_products:
                        continue
                    
                    # Check if it's I-labeled or not in ground truth (both are valid negatives)
                    # We'll accept all non-positive products as hard negatives
                    hard_neg_candidates.append((product_id, float(dist)))
            
            # Take top-k hard negatives
            hard_neg_candidates = sorted(hard_neg_candidates, key=lambda x: x[1])[:top_k_hard]
            
            # Add to hard negatives list
            query_id = query_to_query_id[query]
            split = query_to_split[query]
            
            for product_id, dist in hard_neg_candidates:
                # Hard negatives are used in Stage 2 and Stage 3 (not Stage 1)
                hard_negatives.append({
                    'query': query,
                    'query_id': query_id,
                    'product_id': product_id,
                    'esci_label': 'I',  # Hard negatives are treated as irrelevant
                    'label': 0.0,
                    'weight': 0.0,
                    'neg_type': 'hard',
                    'split': split,
                    'use_in_stage_1': False,  # Stage 1: Random negatives only
                    'use_in_stage_2': True,   # Stage 2: Random + Hard negatives
                    'use_in_stage_3': True    # Stage 3: Random + Hard negatives
                })
    
    hard_neg_df = pd.DataFrame(hard_negatives)
    print(f"\nCreated {len(hard_neg_df):,} hard negative pairs")
    print(f"Average hard negatives per query: {len(hard_neg_df)/len(query_to_relevant):.1f}")
    
    # Print stage distribution
    print(f"\nStage distribution for hard negatives:")
    print(f"  Stage 1: {(hard_neg_df['use_in_stage_1']).sum():,} (not used)")
    print(f"  Stage 2: {(hard_neg_df['use_in_stage_2']).sum():,}")
    print(f"  Stage 3: {(hard_neg_df['use_in_stage_3']).sum():,}")
    
    return hard_neg_df


def add_product_texts(df, product_id_to_text):
    """Add product_text column to dataframe"""
    print("\nAdding product texts...")
    
    def get_product_text(product_id):
        return product_id_to_text.get(product_id, "")
    
    df['product_text'] = df['product_id'].apply(get_product_text)
    
    # Remove rows where product_text is missing (shouldn't happen, but safety check)
    before = len(df)
    df = df[df['product_text'].str.len() > 0].copy()
    after = len(df)
    
    if before != after:
        print(f"   Removed {before - after} rows with missing product text")
    
    return df


def create_splits(positive_df, random_neg_df, hard_neg_df, val_ratio=0.1, seed=42):
    """Create train/val/test splits"""
    print("\n" + "=" * 80)
    print("CREATING DATA SPLITS")
    print("=" * 80)
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Combine all pairs
    all_pairs = pd.concat([positive_df, random_neg_df, hard_neg_df], ignore_index=True)
    print(f"\nTotal pairs: {len(all_pairs):,}")
    print(f"  Positives: {len(positive_df):,}")
    print(f"  Random negatives: {len(random_neg_df):,}")
    print(f"  Hard negatives: {len(hard_neg_df):,}")
    
    # Use existing split from dataset, but create validation from train
    train_pairs = all_pairs[all_pairs['split'] == 'train'].copy()
    test_pairs = all_pairs[all_pairs['split'] == 'test'].copy()
    
    print(f"\nOriginal splits:")
    print(f"  Train: {len(train_pairs):,}")
    print(f"  Test: {len(test_pairs):,}")
    
    # Sample validation set from train
    train_queries = train_pairs['query'].unique()
    n_val_queries = int(len(train_queries) * val_ratio)
    val_queries = set(np.random.choice(train_queries, n_val_queries, replace=False))
    
    val_pairs = train_pairs[train_pairs['query'].isin(val_queries)].copy()
    train_pairs = train_pairs[~train_pairs['query'].isin(val_queries)].copy()
    
    # Update split column
    train_pairs['split'] = 'train'
    val_pairs['split'] = 'val'
    test_pairs['split'] = 'test'
    
    print(f"\nFinal splits:")
    print(f"  Train: {len(train_pairs):,}")
    print(f"  Val: {len(val_pairs):,}")
    print(f"  Test: {len(test_pairs):,}")
    
    # Print statistics per split
    for split_name, split_df in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        print(f"\n{split_name.upper()} split statistics:")
        print(f"  Total: {len(split_df):,}")
        print(f"  Positives: {(split_df['label'] == 1.0).sum():,}")
        print(f"  Negatives: {(split_df['label'] == 0.0).sum():,}")
        print(f"  Random negatives: {(split_df['neg_type'] == 'random').sum():,}")
        print(f"  Hard negatives: {(split_df['neg_type'] == 'hard').sum():,}")
        
        # Print stage distribution
        print(f"  Stage 1 examples: {(split_df['use_in_stage_1']).sum():,}")
        print(f"  Stage 2 examples: {(split_df['use_in_stage_2']).sum():,}")
        print(f"  Stage 3 examples: {(split_df['use_in_stage_3']).sum():,}")
    
    return train_pairs, val_pairs, test_pairs


def save_training_data(train_df, val_df, test_df, output_dir):
    """Save training data to parquet files"""
    print("\n" + "=" * 80)
    print("SAVING TRAINING DATA")
    print("=" * 80)
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save training pairs
    train_file = output_dir / "training_pairs.parquet"
    train_df.to_parquet(train_file, index=False)
    print(f"\n1. Training pairs saved: {train_file}")
    print(f"   Rows: {len(train_df):,}")
    
    # Save validation pairs
    val_file = output_dir / "val_pairs.parquet"
    val_df.to_parquet(val_file, index=False)
    print(f"\n2. Validation pairs saved: {val_file}")
    print(f"   Rows: {len(val_df):,}")
    
    # Save test pairs
    test_file = output_dir / "test_pairs.parquet"
    test_df.to_parquet(test_file, index=False)
    print(f"\n3. Test pairs saved: {test_file}")
    print(f"   Rows: {len(test_df):,}")
    
    # Save statistics
    stats = {
        'train': {
            'total': len(train_df),
            'positives': int((train_df['label'] == 1.0).sum()),
            'negatives': int((train_df['label'] == 0.0).sum()),
            'random_negatives': int((train_df['neg_type'] == 'random').sum()),
            'hard_negatives': int((train_df['neg_type'] == 'hard').sum()),
            'stage_1_examples': int((train_df['use_in_stage_1']).sum()),
            'stage_2_examples': int((train_df['use_in_stage_2']).sum()),
            'stage_3_examples': int((train_df['use_in_stage_3']).sum()),
        },
        'val': {
            'total': len(val_df),
            'positives': int((val_df['label'] == 1.0).sum()),
            'negatives': int((val_df['label'] == 0.0).sum()),
            'random_negatives': int((val_df['neg_type'] == 'random').sum()),
            'hard_negatives': int((val_df['neg_type'] == 'hard').sum()),
            'stage_1_examples': int((val_df['use_in_stage_1']).sum()),
            'stage_2_examples': int((val_df['use_in_stage_2']).sum()),
            'stage_3_examples': int((val_df['use_in_stage_3']).sum()),
        },
        'test': {
            'total': len(test_df),
            'positives': int((test_df['label'] == 1.0).sum()),
            'negatives': int((test_df['label'] == 0.0).sum()),
            'random_negatives': int((test_df['neg_type'] == 'random').sum()),
            'hard_negatives': int((test_df['neg_type'] == 'hard').sum()),
            'stage_1_examples': int((test_df['use_in_stage_1']).sum()),
            'stage_2_examples': int((test_df['use_in_stage_2']).sum()),
            'stage_3_examples': int((test_df['use_in_stage_3']).sum()),
        }
    }
    
    stats_file = output_dir / "training_stats.pkl"
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)
    print(f"\n4. Statistics saved: {stats_file}")
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)


def main():
    """Main function to orchestrate data preparation"""
    print("=" * 80)
    print("TWO-TOWER TRAINING DATA PREPARATION")
    print("=" * 80)
    
    # Step 1: Load data
    (examples_df, products_df, embedding_model, hnsw_index, 
     product_ids, product_id_to_idx, product_id_to_text, embedding_dim) = load_data()
    
    # Step 2: Create positive pairs
    positive_df = create_positive_pairs(examples_df, products_df)
    
    # Step 3: Sample random negatives
    random_neg_df = sample_random_negatives(
        examples_df, products_df, positive_df, 
        neg_ratio=3, seed=42
    )
    
    # Step 4: Generate hard negatives
    hard_neg_df = generate_hard_negatives(
        examples_df, positive_df, embedding_model,
        hnsw_index, product_ids, product_id_to_idx,
        product_id_to_text, k=100, top_k_hard=20, seed=42
    )
    
    # Step 5: Add product texts
    positive_df = add_product_texts(positive_df, product_id_to_text)
    random_neg_df = add_product_texts(random_neg_df, product_id_to_text)
    hard_neg_df = add_product_texts(hard_neg_df, product_id_to_text)
    
    # Step 6: Create splits
    train_df, val_df, test_df = create_splits(
        positive_df, random_neg_df, hard_neg_df, 
        val_ratio=0.1, seed=42
    )
    
    # Step 7: Save everything
    save_training_data(train_df, val_df, test_df, OUTPUT_DIR)
    
    print("\n✓ All done! Training data is ready.")


if __name__ == "__main__":
    main()
