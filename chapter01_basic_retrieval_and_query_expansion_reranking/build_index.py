"""
Script to build FAISS indices from shopping queries dataset

This script:
1. Loads and cleans product data
2. Generates embeddings in chunks (memory-efficient)
3. Builds FAISS indices incrementally
4. Saves everything to disk
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import gc
import os
from pathlib import Path
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()

# Set environment variables early to avoid multiprocessing issues
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# os.environ['OMP_NUM_THREADS'] = '1'

# Set up paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_product_text(row):
    """Combine product fields into a single text for embedding"""
    parts = []
    
    # Title (always present)
    if pd.notna(row['product_title']):
        parts.append(str(row['product_title']).strip())
    
    # Description
    if pd.notna(row['product_description']):
        desc = str(row['product_description']).strip()
        if desc and desc.lower() != 'none':
            parts.append(desc)
    
    # Bullet points
    if pd.notna(row['product_bullet_point']):
        bullets = str(row['product_bullet_point']).strip()
        if bullets and bullets.lower() != 'none':
            parts.append(bullets)
    
    # Brand
    if pd.notna(row['product_brand']):
        brand = str(row['product_brand']).strip()
        if brand and brand.lower() != 'none':
            parts.append(f"Brand: {brand}")
    
    # Color
    if pd.notna(row['product_color']):
        color = str(row['product_color']).strip()
        if color and color.lower() != 'none':
            parts.append(f"Color: {color}")
    
    # Join all parts
    combined = " ".join(parts)
    
    # Clean up multiple spaces
    combined = " ".join(combined.split())
    
    return combined if combined else "No description available"


def load_and_clean_data():
    """Load and clean the product dataset"""
    print("=" * 80)
    print("LOADING AND CLEANING DATA")
    print("=" * 80)
    
    # Load products dataset
    print("\n1. Loading products dataset...")
    products_df = pd.read_parquet(DATA_DIR / "shopping_queries_dataset_products.parquet")
    print(f"   Loaded {len(products_df):,} products")
    
    # Load examples to get product IDs we need
    print("\n2. Loading examples dataset to filter products...")
    examples_df = pd.read_parquet(DATA_DIR / "shopping_queries_dataset_examples.parquet")
    products_in_examples = set(examples_df['product_id'].unique())
    print(f"   Found {len(products_in_examples):,} unique products in examples")
    
    # Filter products
    print("\n3. Filtering products...")
    products_df_clean = products_df[products_df['product_id'].isin(products_in_examples)].copy()
    print(f"   Filtered to {len(products_df_clean):,} products")
    
    # Create combined text field
    print("\n4. Creating combined text field...")
    products_df_clean['product_text'] = products_df_clean.progress_apply(create_product_text, axis=1)
    
    # Remove products with empty text
    products_df_clean = products_df_clean[products_df_clean['product_text'].str.len() > 0].copy()
    print(f"   Final product count: {len(products_df_clean):,}")
    
    return products_df_clean


def build_indices_incremental(products_df_clean, embedding_model, embedding_dim):
    """Build FAISS indices incrementally by processing data in chunks"""
    print("\n" + "=" * 80)
    print("BUILDING FAISS INDICES (INCREMENTAL)")
    print("=" * 80)
    
    n_products = len(products_df_clean)
    chunk_size = 100000  # Smaller chunks to avoid memory issues
    batch_size = 4  # Smaller batch size for embedding model
    
    print(f"\nTotal products: {n_products:,}")
    print(f"Chunk size: {chunk_size:,} products")
    print(f"Embedding batch size: {batch_size}")
    
    # Initialize FAISS indices
    print("\nInitializing FAISS indices...")
    flat_index = faiss.IndexFlatL2(embedding_dim)
    
    # HNSW index
    m = 32
    ef_construction = 200
    ef_search = 50
    hnsw_index = faiss.IndexHNSWFlat(embedding_dim, m)
    hnsw_index.hnsw.efConstruction = ef_construction
    hnsw_index.hnsw.efSearch = ef_search
    
    # Process dataframe in chunks
    print(f"\nProcessing embeddings and building indices...")
    n_chunks = (n_products + chunk_size - 1) // chunk_size
    
    # Store product IDs as we process
    product_ids = []
    
    for chunk_idx in range(n_chunks):
        try:
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_products)
            
            # Get chunk directly from dataframe
            chunk_df = products_df_clean.iloc[start_idx:end_idx].copy()
            chunk_texts = chunk_df['product_text'].tolist()
            chunk_ids = chunk_df['product_id'].tolist()
            product_ids.extend(chunk_ids)
            
            # Generate embeddings for this chunk
            # Disable multiprocessing to avoid segfaults
            print(f"Generating embeddings for chunk {chunk_idx}...")
            with torch.no_grad():
                chunk_embeddings = embedding_model.encode(
                    chunk_texts,
                    batch_size=batch_size
                )
            
            # Add to Flat index
            print(f"Adding to Flat index...")
            flat_index.add(chunk_embeddings)
            
            # Add to HNSW index
            print(f"Adding to HNSW index...")
            hnsw_index.add(chunk_embeddings)
            
            # Clean up memory
            del chunk_df, chunk_texts, chunk_embeddings
            gc.collect()
            
            # Print progress every 20 chunks
            if (chunk_idx + 1) % 20 == 0:
                print(f"   Processed {end_idx:,} / {n_products:,} products ({end_idx/n_products*100:.1f}%)")
        
        except Exception as e:
            print(f"\nError processing chunk {chunk_idx}: {e}")
            print(f"Chunk range: {start_idx} to {end_idx}")
            raise
    
    print(f"\nAll indices built successfully!")
    print(f"  Flat index: {flat_index.ntotal:,} vectors")
    print(f"  HNSW index: {hnsw_index.ntotal:,} vectors (m={m})")
    
    return flat_index, hnsw_index, product_ids, m, ef_construction, ef_search


def save_indices(flat_index, hnsw_index, product_ids, products_df_clean, 
                 embedding_dim, model_name, m, ef_construction, ef_search):
    """Save all indices and metadata to disk"""
    print("\n" + "=" * 80)
    print("SAVING TO DISK")
    print("=" * 80)
    
    # 1. Save FAISS indices
    print("\n1. Saving FAISS indices...")
    
    flat_index_file = OUTPUT_DIR / "faiss_index_flat.bin"
    faiss.write_index(flat_index, str(flat_index_file))
    print(f"   Flat index saved to: {flat_index_file}")
    
    hnsw_index_file = OUTPUT_DIR / "faiss_index_hnsw.bin"
    faiss.write_index(hnsw_index, str(hnsw_index_file))
    print(f"   HNSW index saved to: {hnsw_index_file}")
    
    # 2. Save product IDs mapping
    print("\n2. Saving product IDs mapping...")
    product_ids_file = OUTPUT_DIR / "product_ids.pkl"
    with open(product_ids_file, 'wb') as f:
        pickle.dump(product_ids, f)
    print(f"   Product IDs saved to: {product_ids_file}")
    
    # 3. Save cleaned products dataframe
    print("\n3. Saving cleaned products dataframe...")
    products_file = OUTPUT_DIR / "products_clean.parquet"
    products_df_clean.to_parquet(products_file, index=False)
    print(f"   Products dataframe saved to: {products_file}")
    
    # 4. Save metadata
    print("\n4. Saving metadata...")
    metadata = {
        'embedding_dim': embedding_dim,
        'model_name': model_name,
        'n_products': len(product_ids),
        'index_types': ['flat', 'hnsw'],
        'hnsw_params': {'m': m, 'ef_construction': ef_construction, 'ef_search': ef_search}
    }
    
    metadata_file = OUTPUT_DIR / "metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   Metadata saved to: {metadata_file}")
    
    print("\n" + "=" * 80)
    print("ALL DATA SAVED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")


def main():
    """Main function to orchestrate the entire process"""
    print("=" * 80)
    print("FAISS INDEX BUILDER - Shopping Queries Dataset")
    print("=" * 80)
    
    # Step 1: Load and clean data
    products_df_clean = load_and_clean_data()
    
    # Step 2: Initialize embedding model
    print("\n" + "=" * 80)
    print("INITIALIZING EMBEDDING MODEL")
    print("=" * 80)
    model_name = "all-MiniLM-L6-v2"
    print(f"Loading model: {model_name}")
    
    embedding_model = SentenceTransformer(model_name)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    print(f"Model loaded! Embedding dimension: {embedding_dim}")
    
    # Step 3: Build indices incrementally
    flat_index, hnsw_index, product_ids, m, ef_construction, ef_search = \
        build_indices_incremental(products_df_clean, embedding_model, embedding_dim)
    
    # Step 4: Save everything
    save_indices(flat_index, hnsw_index, product_ids, products_df_clean,
                embedding_dim, model_name, m, ef_construction, ef_search)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nDataset Statistics:")
    print(f"  Products processed: {len(products_df_clean):,}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Estimated embedding size: {flat_index.ntotal * embedding_dim * 4 / 1024 / 1024:.2f} MB")
    print(f"\nIndex Statistics:")
    print(f"  Flat index: {flat_index.ntotal:,} vectors")
    print(f"  HNSW index: {hnsw_index.ntotal:,} vectors")
    print("\n✓ Process complete!")


if __name__ == "__main__":
    main()
