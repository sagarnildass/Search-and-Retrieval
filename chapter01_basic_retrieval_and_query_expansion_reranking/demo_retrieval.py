"""
End-to-End Demo Script for FAISS Retrieval System

This script demonstrates how to use the built FAISS indices for product retrieval.
It shows example queries and displays retrieved results.
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Set up paths
ROOT_DIR = Path(__file__).parent.parent
INDEX_DIR = Path(__file__).parent / "data"


class RetrievalDemo:
    """Demo class for FAISS retrieval system"""
    
    def __init__(self, index_dir=INDEX_DIR):
        """Initialize the retrieval demo"""
        self.index_dir = Path(index_dir)
        self.flat_index = None
        self.hnsw_index = None
        self.embedding_model = None
        self.product_ids = None
        self.products_df = None
        self.embedding_dim = None
        
    def load(self):
        """Load indices, model, and data"""
        print("=" * 80)
        print("LOADING RETRIEVAL SYSTEM")
        print("=" * 80)
        
        # Load metadata
        print("\n1. Loading metadata...")
        with open(self.index_dir / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.embedding_dim = metadata['embedding_dim']
        model_name = metadata['model_name']
        
        print(f"   Embedding dimension: {self.embedding_dim}")
        print(f"   Model: {model_name}")
        print(f"   Number of products: {metadata['n_products']:,}")
        
        # Load embedding model
        print("\n2. Loading embedding model...")
        self.embedding_model = SentenceTransformer(model_name)
        print(f"   Model loaded: {model_name}")
        
        # Load FAISS indices
        print("\n3. Loading FAISS indices...")
        
        self.flat_index = faiss.read_index(str(self.index_dir / "faiss_index_flat.bin"))
        print(f"   Flat index loaded: {self.flat_index.ntotal:,} vectors")
        
        self.hnsw_index = faiss.read_index(str(self.index_dir / "faiss_index_hnsw.bin"))
        # Set optimal ef_search for better recall (based on evaluation results)
        self.hnsw_index.hnsw.efSearch = 200
        print(f"   HNSW index loaded: {self.hnsw_index.ntotal:,} vectors")
        print(f"   HNSW ef_search set to: {self.hnsw_index.hnsw.efSearch} (optimal for recall)")
        
        # Load product IDs
        print("\n4. Loading product IDs...")
        with open(self.index_dir / "product_ids.pkl", 'rb') as f:
            self.product_ids = pickle.load(f)
        print(f"   Loaded {len(self.product_ids):,} product IDs")
        
        # Load products dataframe
        print("\n5. Loading products dataframe...")
        self.products_df = pd.read_parquet(self.index_dir / "products_clean.parquet")
        print(f"   Loaded {len(self.products_df):,} products")
        
        print("\nRetrieval system loaded successfully!")
    
    def search(self, query, k=20, index_type='hnsw', ef_search=None):
        """
        Search for products
        
        Args:
            query: Query text string
            k: Number of results to return (default: 20 for better recall)
            index_type: 'flat' or 'hnsw'
            ef_search: For HNSW, override ef_search parameter (default: 200, optimal)
        
        Returns:
            List of tuples (product_id, product_title, distance, search_time_ms)
        """
        if index_type not in ['flat', 'hnsw']:
            raise ValueError("index_type must be 'flat' or 'hnsw'")
        
        index = self.flat_index if index_type == 'flat' else self.hnsw_index
        
        # Set ef_search for HNSW if specified
        original_ef_search = None
        if index_type == 'hnsw' and ef_search is not None:
            original_ef_search = index.hnsw.efSearch
            index.hnsw.efSearch = ef_search
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].astype('float32')
        
        # Search
        start_time = time.time()
        distances, indices = index.search(query_embedding.reshape(1, -1), k)
        search_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Restore original ef_search if changed
        if original_ef_search is not None:
            index.hnsw.efSearch = original_ef_search
        
        # Get results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.product_ids):
                product_id = self.product_ids[idx]
                product_row = self.products_df[self.products_df['product_id'] == product_id]
                
                if len(product_row) > 0:
                    product_title = product_row.iloc[0]['product_title']
                    results.append((product_id, product_title, float(dist), search_time))
        
        return results
    
    def display_results(self, query, results, index_type, k):
        """Display search results in a formatted way"""
        print(f"\n{'=' * 80}")
        print(f"QUERY: '{query}'")
        print(f"Index: {index_type.upper()} | Top {k} Results")
        print(f"{'=' * 80}")
        
        if not results:
            print("No results found.")
            return
        
        search_time = results[0][3]  # Search time is same for all results
        
        print(f"\nSearch time: {search_time:.2f}ms")
        if index_type.lower() == 'hnsw':
            print(f"ef_search: {self.hnsw_index.hnsw.efSearch}")
        print(f"\nResults:")
        print("-" * 80)
        
        for i, (product_id, title, distance, _) in enumerate(results, 1):
            print(f"\n{i}. [{product_id}]")
            print(f"   Title: {title[:100]}{'...' if len(title) > 100 else ''}")
            print(f"   Distance: {distance:.4f}")
    
    def compare_indices(self, query, k=20):
        """Compare results from both indices"""
        print(f"\n{'=' * 80}")
        print(f"COMPARING INDICES FOR QUERY: '{query}'")
        print(f"{'=' * 80}")
        
        # Search in both indices
        flat_results = self.search(query, k=k, index_type='flat')
        hnsw_results = self.search(query, k=k, index_type='hnsw')
        
        # Display Flat results
        self.display_results(query, flat_results, 'flat', k)
        
        # Display HNSW results
        self.display_results(query, hnsw_results, 'hnsw', k)
        
        # Compare
        print(f"\n{'=' * 80}")
        print("COMPARISON")
        print(f"{'=' * 80}")
        
        flat_time = flat_results[0][3] if flat_results else 0
        hnsw_time = hnsw_results[0][3] if hnsw_results else 0
        
        print(f"\nSpeed:")
        print(f"  Flat:  {flat_time:.2f}ms")
        print(f"  HNSW:  {hnsw_time:.2f}ms")
        if hnsw_time > 0:
            print(f"  Speedup: {flat_time/hnsw_time:.2f}x")
        
        # Check overlap
        flat_ids = {r[0] for r in flat_results}
        hnsw_ids = {r[0] for r in hnsw_results}
        overlap = flat_ids.intersection(hnsw_ids)
        
        print(f"\nResult Overlap:")
        print(f"  Common results: {len(overlap)}/{k} ({len(overlap)/k*100:.1f}%)")
        
        if overlap != flat_ids:
            print(f"\n  Results only in Flat: {flat_ids - hnsw_ids}")
        if overlap != hnsw_ids:
            print(f"\n  Results only in HNSW: {hnsw_ids - flat_ids}")


def main():
    """Main demo function"""
    print("=" * 80)
    print("FAISS RETRIEVAL SYSTEM DEMO")
    print("=" * 80)
    
    # Initialize and load
    demo = RetrievalDemo()
    demo.load()
    
    # Example queries
    example_queries = [
        "wireless bluetooth headphones",
        "laptop computer",
        "running shoes",
        "coffee maker",
        "smartphone case"
    ]
    
    print("\n" + "=" * 80)
    print("DEMO: SINGLE INDEX SEARCH")
    print("=" * 80)
    
    # Demo with HNSW index (faster, optimal settings)
    print("\nUsing HNSW index (ef_search=200, optimal for recall)...")
    for query in example_queries[:3]:  # Show first 3 queries
        results = demo.search(query, k=10, index_type='hnsw')
        demo.display_results(query, results, 'hnsw', k=10)
    
    print("\n" + "=" * 80)
    print("DEMO: COMPARING BOTH INDICES")
    print("=" * 80)
    
    # Compare indices for a sample query
    sample_query = example_queries[0]
    demo.compare_indices(sample_query, k=20)
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("\nYou can now search for products!")
    print("Enter queries (or 'quit' to exit):")
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print("\nChoose index:")
            print("  1. Flat (exact, slower, ~184ms)")
            print("  2. HNSW (approximate, faster, ~2ms, ef_search=200)")
            print("  3. Compare both")
            print("  4. HNSW with custom K (enter number)")
            
            choice = input("Choice (1/2/3/4): ").strip()
            
            if choice == '1':
                results = demo.search(query, k=20, index_type='flat')
                demo.display_results(query, results, 'flat', k=20)
            elif choice == '2':
                results = demo.search(query, k=20, index_type='hnsw')
                demo.display_results(query, results, 'hnsw', k=20)
            elif choice == '3':
                demo.compare_indices(query, k=20)
            elif choice == '4':
                try:
                    k_val = int(input("Enter K (number of results, e.g., 50, 100, 200): "))
                    results = demo.search(query, k=k_val, index_type='hnsw')
                    demo.display_results(query, results, 'hnsw', k=k_val)
                except ValueError:
                    print("Invalid number. Using default K=20.")
                    results = demo.search(query, k=20, index_type='hnsw')
                    demo.display_results(query, results, 'hnsw', k=20)
            else:
                print("Invalid choice. Using HNSW with K=20 by default.")
                results = demo.search(query, k=20, index_type='hnsw')
                demo.display_results(query, results, 'hnsw', k=20)
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
