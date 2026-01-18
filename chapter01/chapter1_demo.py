"""
Chapter 1: Complete Demo - Basic Vector Retrieval with FAISS

This script demonstrates the complete workflow:
1. Creating/generating product data
2. Building FAISS indices (Flat, IVF, HNSW)
3. Performing searches
4. Evaluating retrieval performance
5. Comparing different index types
"""

import time
import numpy as np
from chapter1_basic_retrieval import (
    RetrievalSystem,
    Product,
    create_sample_products
)
from chapter1_evaluation import (
    RetrievalEvaluator,
    create_sample_ground_truth
)


def compare_index_types(products: list, queries: list, ground_truth: dict):
    """
    Compare different FAISS index types on performance and accuracy
    
    Args:
        products: List of Product objects
        queries: List of query strings
        ground_truth: Dictionary mapping query to set of relevant product IDs
    """
    print("\n" + "=" * 80)
    print("COMPARING INDEX TYPES")
    print("=" * 80)
    
    index_configs = [
        {
            'type': 'flat',
            'name': 'FlatL2 (Exact)',
            'kwargs': {}
        },
        {
            'type': 'ivf',
            'name': 'IVF (Approximate)',
            'kwargs': {'nlist': 100, 'nprobe': 10}
        },
        {
            'type': 'hnsw',
            'name': 'HNSW (Approximate)',
            'kwargs': {'m': 32, 'ef_construction': 200, 'ef_search': 50}
        }
    ]
    
    results = []
    
    for config in index_configs:
        print(f"\n{config['name']}:")
        print("-" * 80)
        
        # Build index
        system = RetrievalSystem(index_type=config['type'])
        system.add_products(products)
        
        build_start = time.time()
        system.build_index(**config['kwargs'])
        build_time = time.time() - build_start
        
        # Perform searches
        query_results = []
        search_times = []
        
        search_start = time.time()
        for query in queries:
            query_start = time.time()
            search_results = system.search(query, k=20)
            query_time = time.time() - query_start
            search_times.append(query_time)
            
            retrieved_ids = [product.id for product, _ in search_results]
            query_results.append((query, retrieved_ids))
        
        total_search_time = time.time() - search_start
        avg_search_time = np.mean(search_times)
        
        # Evaluate
        eval_result = RetrievalEvaluator.evaluate(
            query_results,
            ground_truth,
            k_values=[1, 5, 10, 20],
            compute_ndcg=True,
            compute_map=True
        )
        
        results.append({
            'name': config['name'],
            'build_time': build_time,
            'avg_search_time': avg_search_time,
            'total_search_time': total_search_time,
            'evaluation': eval_result
        })
        
        print(f"  Build time: {build_time:.4f}s")
        print(f"  Average search time: {avg_search_time*1000:.2f}ms")
        print(f"  Total search time ({len(queries)} queries): {total_search_time:.4f}s")
        print(f"  Recall@10: {eval_result.recall_at_k[10]:.4f}")
        print(f"  Precision@10: {eval_result.precision_at_k[10]:.4f}")
        print(f"  MRR: {eval_result.mrr:.4f}")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"{'Index Type':<20} {'Build(s)':<12} {'Search(ms)':<12} {'Recall@10':<12} {'MRR':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<20} {r['build_time']:<12.4f} {r['avg_search_time']*1000:<12.2f} "
              f"{r['evaluation'].recall_at_k[10]:<12.4f} {r['evaluation'].mrr:<12.4f}")


def main():
    """Main demo function"""
    print("=" * 80)
    print("CHAPTER 1: BASIC VECTOR RETRIEVAL WITH FAISS - COMPLETE DEMO")
    print("=" * 80)
    
    # Step 1: Create sample products
    print("\n[Step 1] Creating sample product catalog...")
    n_products = 5000
    products = create_sample_products(n_products)
    print(f"Created {len(products)} products")
    print(f"Sample product: {products[0]}")
    
    # Step 2: Create test queries
    print("\n[Step 2] Creating test queries...")
    test_queries = [
        "electronics gadgets",
        "books about science",
        "sports equipment",
        "kitchen appliances",
        "beauty products",
        "toys for children",
        "automotive parts",
        "home decor items"
    ]
    print(f"Created {len(test_queries)} test queries")
    
    # Step 3: Create ground truth (in practice, this would come from real labels)
    print("\n[Step 3] Creating ground truth labels...")
    all_product_ids = [p.id for p in products]
    ground_truth = create_sample_ground_truth(test_queries, all_product_ids, seed=42)
    print(f"Created ground truth for {len(ground_truth)} queries")
    
    # Step 4: Build and test Flat index
    print("\n[Step 4] Building FlatL2 index (exact search)...")
    flat_system = RetrievalSystem(index_type="flat")
    flat_system.add_products(products)
    
    build_start = time.time()
    flat_system.build_index()
    build_time = time.time() - build_start
    print(f"Index built in {build_time:.4f} seconds")
    
    # Step 5: Perform searches
    print("\n[Step 5] Performing searches...")
    print("-" * 80)
    for query in test_queries[:3]:  # Show first 3 queries
        print(f"\nQuery: '{query}'")
        start_time = time.time()
        results = flat_system.search(query, k=5)
        search_time = time.time() - start_time
        
        print(f"Search time: {search_time*1000:.2f}ms")
        print("Top 5 results:")
        for i, (product, distance) in enumerate(results, 1):
            print(f"  {i}. [{product.id}] {product.title}")
            print(f"     Category: {product.category}, Distance: {distance:.4f}")
    
    # Step 6: Evaluate retrieval performance
    print("\n[Step 6] Evaluating retrieval performance...")
    print("-" * 80)
    
    query_results = []
    for query in test_queries:
        results = flat_system.search(query, k=20)
        retrieved_ids = [product.id for product, _ in results]
        query_results.append((query, retrieved_ids))
    
    eval_result = RetrievalEvaluator.evaluate(
        query_results,
        ground_truth,
        k_values=[1, 5, 10, 20],
        compute_ndcg=True,
        compute_map=True
    )
    
    print(eval_result)
    
    # Step 7: Compare different index types
    compare_index_types(products, test_queries, ground_truth)
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. FlatL2 provides exact search but is slower for large datasets")
    print("2. IVF and HNSW provide approximate search with better speed")
    print("3. Trade-off between recall/accuracy and search latency")
    print("4. Choose index type based on dataset size and latency requirements")


if __name__ == "__main__":
    main()
