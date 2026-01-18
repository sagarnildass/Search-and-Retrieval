"""
End-to-End Evaluation Script for FAISS Indices

This script evaluates Flat and HNSW indices using ground truth labels.
It calculates Recall@K, Precision@K, MRR, NDCG@K, and MAP metrics.

Ground truth format: {query_text: set(relevant_product_ids)}
- Relevant products are those with ESCI labels: E (Exact), S (Substitute), C (Complement)
- Irrelevant products have label: I (Irrelevant)
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Set up paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR
INDEX_DIR = Path(__file__).parent / "data"
GROUND_TRUTH_FILE = Path(__file__).parent / "ground_truth.pkl"


# Evaluation Metrics Classes and Functions
class RetrievalEvaluator:
    """Evaluates retrieval systems using standard IR metrics"""
    
    @staticmethod
    def recall_at_k(retrieved_ids, relevant_ids, k):
        """Calculate Recall@K"""
        if len(relevant_ids) == 0:
            return 0.0
        top_k_retrieved = set(retrieved_ids[:k])
        relevant_retrieved = top_k_retrieved.intersection(relevant_ids)
        return len(relevant_retrieved) / len(relevant_ids)
    
    @staticmethod
    def precision_at_k(retrieved_ids, relevant_ids, k):
        """Calculate Precision@K"""
        if k == 0:
            return 0.0
        top_k_retrieved = set(retrieved_ids[:k])
        relevant_retrieved = top_k_retrieved.intersection(relevant_ids)
        return len(relevant_retrieved) / k
    
    @staticmethod
    def mean_reciprocal_rank(query_results, ground_truth):
        """Calculate Mean Reciprocal Rank (MRR)"""
        if len(query_results) == 0:
            return 0.0
        reciprocal_ranks = []
        for query_id, retrieved_ids in query_results:
            relevant_ids = ground_truth.get(query_id, set())
            if len(relevant_ids) == 0:
                continue
            rank = None
            for i, item_id in enumerate(retrieved_ids, start=1):
                if item_id in relevant_ids:
                    rank = i
                    break
            if rank is not None:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_ids, relevant_ids, k, relevance_scores=None):
        """Calculate Normalized Discounted Cumulative Gain@K"""
        if relevance_scores is None:
            relevance_scores = {item_id: 1.0 if item_id in relevant_ids else 0.0
                              for item_id in retrieved_ids[:k]}
        dcg = 0.0
        for i, item_id in enumerate(retrieved_ids[:k], start=1):
            rel = relevance_scores.get(item_id, 0.0)
            dcg += rel / np.log2(i + 1)
        ideal_relevance = sorted(
            [relevance_scores.get(item_id, 0.0) for item_id in retrieved_ids[:k]],
            reverse=True
        )
        idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal_relevance, start=1))
        if idcg == 0.0:
            return 0.0
        return dcg / idcg
    
    @staticmethod
    def mean_average_precision(query_results, ground_truth):
        """Calculate Mean Average Precision (MAP)"""
        if len(query_results) == 0:
            return 0.0
        average_precisions = []
        for query_id, retrieved_ids in query_results:
            relevant_ids = ground_truth.get(query_id, set())
            if len(relevant_ids) == 0:
                continue
            precisions = []
            relevant_count = 0
            for i, item_id in enumerate(retrieved_ids, start=1):
                if item_id in relevant_ids:
                    relevant_count += 1
                    precision_at_i = relevant_count / i
                    precisions.append(precision_at_i)
            if precisions:
                ap = np.mean(precisions)
                average_precisions.append(ap)
        return np.mean(average_precisions) if average_precisions else 0.0
    
    @staticmethod
    def evaluate(query_results, ground_truth, k_values=[1, 5, 10, 20], 
                 compute_ndcg=True, compute_map=True):
        """Comprehensive evaluation of retrieval results"""
        recall_scores = {}
        precision_scores = {}
        ndcg_scores = {} if compute_ndcg else None
        
        for k in k_values:
            recalls = []
            precisions = []
            ndcgs = []
            for query_id, retrieved_ids in query_results:
                relevant_ids = ground_truth.get(query_id, set())
                recall = RetrievalEvaluator.recall_at_k(retrieved_ids, relevant_ids, k)
                precision = RetrievalEvaluator.precision_at_k(retrieved_ids, relevant_ids, k)
                recalls.append(recall)
                precisions.append(precision)
                if compute_ndcg:
                    ndcg = RetrievalEvaluator.ndcg_at_k(retrieved_ids, relevant_ids, k)
                    ndcgs.append(ndcg)
            recall_scores[k] = np.mean(recalls)
            precision_scores[k] = np.mean(precisions)
            if compute_ndcg:
                ndcg_scores[k] = np.mean(ndcgs)
        
        mrr = RetrievalEvaluator.mean_reciprocal_rank(query_results, ground_truth)
        map_score = None
        if compute_map:
            map_score = RetrievalEvaluator.mean_average_precision(query_results, ground_truth)
        
        return EvaluationResult(
            recall_at_k=recall_scores,
            precision_at_k=precision_scores,
            mrr=mrr,
            ndcg_at_k=ndcg_scores,
            map_score=map_score
        )


class EvaluationResult:
    """Container for evaluation metrics"""
    def __init__(self, recall_at_k, precision_at_k, mrr, ndcg_at_k=None, map_score=None):
        self.recall_at_k = recall_at_k
        self.precision_at_k = precision_at_k
        self.mrr = mrr
        self.ndcg_at_k = ndcg_at_k
        self.map_score = map_score
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        result = {
            'recall_at_k': self.recall_at_k,
            'precision_at_k': self.precision_at_k,
            'mrr': self.mrr
        }
        if self.ndcg_at_k:
            result['ndcg_at_k'] = self.ndcg_at_k
        if self.map_score is not None:
            result['map_score'] = self.map_score
        return result


def load_indices_and_data():
    """Load FAISS indices, product data, and ground truth"""
    print("=" * 80)
    print("LOADING INDICES AND DATA")
    print("=" * 80)
    
    # Load metadata
    print("\n1. Loading metadata...")
    with open(INDEX_DIR / "metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    print(f"   Embedding dimension: {metadata['embedding_dim']}")
    print(f"   Model: {metadata['model_name']}")
    print(f"   Number of products: {metadata['n_products']:,}")
    
    embedding_dim = metadata['embedding_dim']
    model_name = metadata['model_name']
    
    # Load embedding model
    print("\n2. Loading embedding model...")
    embedding_model = SentenceTransformer(model_name)
    print(f"   Model loaded: {model_name}")
    
    # Load FAISS indices
    print("\n3. Loading FAISS indices...")
    
    flat_index = faiss.read_index(str(INDEX_DIR / "faiss_index_flat.bin"))
    print(f"   Flat index loaded: {flat_index.ntotal:,} vectors")
    
    hnsw_index = faiss.read_index(str(INDEX_DIR / "faiss_index_hnsw.bin"))
    print(f"   HNSW index loaded: {hnsw_index.ntotal:,} vectors")
    
    # Load product IDs
    print("\n4. Loading product IDs...")
    with open(INDEX_DIR / "product_ids.pkl", 'rb') as f:
        product_ids = pickle.load(f)
    print(f"   Loaded {len(product_ids):,} product IDs")
    
    # Load products dataframe
    print("\n5. Loading products dataframe...")
    products_df = pd.read_parquet(INDEX_DIR / "products_clean.parquet")
    print(f"   Loaded {len(products_df):,} products")
    
    # Load ground truth
    print("\n6. Loading ground truth...")
    with open(GROUND_TRUTH_FILE, 'rb') as f:
        ground_truth = pickle.load(f)
    print(f"   Loaded ground truth for {len(ground_truth):,} queries")
    print(f"   Average relevant products per query: {np.mean([len(v) for v in ground_truth.values()]):.1f}")
    
    return (flat_index, hnsw_index, embedding_model, product_ids, 
            products_df, ground_truth, embedding_dim)


def evaluate_index(index, index_name, queries, ground_truth, embedding_model, 
                   product_ids, k_values=[1, 5, 10, 20, 50, 100, 200], ef_search=None):
    """
    Evaluate a single FAISS index
    
    Args:
        index: FAISS index
        index_name: Name of the index ('flat' or 'hnsw')
        queries: List of query strings
        ground_truth: Dictionary mapping query -> set of relevant product IDs
        embedding_model: Sentence transformer model
        product_ids: List of product IDs corresponding to index positions
        k_values: List of K values for evaluation metrics
        ef_search: For HNSW index, set ef_search parameter (None uses default)
    
    Returns:
        Tuple of (eval_result, avg_search_time, total_search_time, query_results)
    """
    print(f"\n{'=' * 80}")
    print(f"EVALUATING {index_name.upper()} INDEX")
    if index_name.lower() == 'hnsw' and ef_search:
        print(f"Using ef_search = {ef_search}")
    print(f"{'=' * 80}")
    
    # Set ef_search for HNSW if specified
    original_ef_search = None
    if index_name.lower() == 'hnsw' and ef_search is not None:
        original_ef_search = index.hnsw.efSearch
        index.hnsw.efSearch = ef_search
        print(f"  Set HNSW ef_search to {ef_search} (was {original_ef_search})")
    
    query_results = []
    search_times = []
    
    print(f"\nProcessing {len(queries):,} queries...")
    
    for query in tqdm(queries, desc=f"Evaluating {index_name}"):
        # Generate query embedding
        query_embedding = embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].astype('float32')
        
        # Search
        start_time = time.time()
        max_k = max(k_values)
        distances, indices = index.search(query_embedding.reshape(1, -1), max_k)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        # Get product IDs (handle invalid indices)
        retrieved_ids = []
        for idx in indices[0]:
            if 0 <= idx < len(product_ids):
                retrieved_ids.append(product_ids[idx])
        
        query_results.append((query, retrieved_ids))
    
    # Restore original ef_search if changed
    if original_ef_search is not None:
        index.hnsw.efSearch = original_ef_search
    
    # Calculate metrics
    print(f"\nCalculating metrics...")
    eval_result = RetrievalEvaluator.evaluate(
        query_results,
        ground_truth,
        k_values=k_values,
        compute_ndcg=True,
        compute_map=True
    )
    
    avg_search_time = np.mean(search_times)
    total_search_time = np.sum(search_times)
    
    return eval_result, avg_search_time, total_search_time, query_results


def print_evaluation_results(eval_result, index_name, avg_search_time, total_search_time):
    """Print evaluation results in a formatted way"""
    print(f"\n{'=' * 80}")
    print(f"{index_name.upper()} INDEX RESULTS")
    print(f"{'=' * 80}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Average search time: {avg_search_time*1000:.2f}ms")
    print(f"  Total search time: {total_search_time:.2f}s")
    
    print(f"\nRetrieval Metrics:")
    print(f"  MRR: {eval_result.mrr:.4f}")
    if eval_result.map_score is not None:
        print(f"  MAP: {eval_result.map_score:.4f}")
    
    print(f"\nRecall@K:")
    for k in sorted(eval_result.recall_at_k.keys()):
        print(f"  Recall@{k}: {eval_result.recall_at_k[k]:.4f}")
    
    print(f"\nPrecision@K:")
    for k in sorted(eval_result.precision_at_k.keys()):
        print(f"  Precision@{k}: {eval_result.precision_at_k[k]:.4f}")
    
    if eval_result.ndcg_at_k:
        print(f"\nNDCG@K:")
        for k in sorted(eval_result.ndcg_at_k.keys()):
            print(f"  NDCG@{k}: {eval_result.ndcg_at_k[k]:.4f}")


def compare_indices(flat_results, hnsw_results, flat_time, hnsw_time):
    """Compare results between Flat and HNSW indices"""
    print(f"\n{'=' * 80}")
    print("COMPARISON: FLAT vs HNSW")
    print(f"{'=' * 80}")
    
    print(f"\n{'Metric':<20} {'Flat':<15} {'HNSW':<15} {'Difference':<15}")
    print("-" * 80)
    
    # MRR
    print(f"{'MRR':<20} {flat_results.mrr:<15.4f} {hnsw_results.mrr:<15.4f} "
          f"{hnsw_results.mrr - flat_results.mrr:<15.4f}")
    
    # MAP
    if flat_results.map_score is not None and hnsw_results.map_score is not None:
        print(f"{'MAP':<20} {flat_results.map_score:<15.4f} {hnsw_results.map_score:<15.4f} "
              f"{hnsw_results.map_score - flat_results.map_score:<15.4f}")
    
    # Recall@10
    print(f"{'Recall@10':<20} {flat_results.recall_at_k[10]:<15.4f} "
          f"{hnsw_results.recall_at_k[10]:<15.4f} "
          f"{hnsw_results.recall_at_k[10] - flat_results.recall_at_k[10]:<15.4f}")
    
    # Precision@10
    print(f"{'Precision@10':<20} {flat_results.precision_at_k[10]:<15.4f} "
          f"{hnsw_results.precision_at_k[10]:<15.4f} "
          f"{hnsw_results.precision_at_k[10] - flat_results.precision_at_k[10]:<15.4f}")
    
    # Speed comparison
    speedup = flat_time / hnsw_time if hnsw_time > 0 else 0
    print(f"\nSpeed Comparison:")
    print(f"  Average search time:")
    print(f"    Flat:  {flat_time*1000:.2f}ms")
    print(f"    HNSW:  {hnsw_time*1000:.2f}ms")
    print(f"    HNSW is {speedup:.2f}x faster than Flat")


def main():
    """Main evaluation function"""
    print("=" * 80)
    print("FAISS INDICES EVALUATION")
    print("=" * 80)
    
    # Load everything
    (flat_index, hnsw_index, embedding_model, product_ids, 
     products_df, ground_truth, embedding_dim) = load_indices_and_data()
    
    # Get test queries
    print("\n" + "=" * 80)
    print("PREPARING TEST QUERIES")
    print("=" * 80)
    
    # Use all queries from ground truth
    test_queries = list(ground_truth.keys())
    print(f"\nTotal queries in ground truth: {len(test_queries):,}")
    
    # Optionally sample for faster evaluation
    # Set sample_size to a number (e.g., 1000) to sample, or None to use all
    sample_size = 1000  # Change to a number like 1000 for faster evaluation
    
    if sample_size and sample_size < len(test_queries):
        print(f"Sampling {sample_size:,} queries for evaluation...")
        np.random.seed(42)
        test_queries = np.random.choice(test_queries, sample_size, replace=False).tolist()
        print(f"Using {len(test_queries):,} queries for evaluation")
    
    # Create subset of ground truth for sampled queries
    test_ground_truth = {q: ground_truth[q] for q in test_queries if q in ground_truth}
    print(f"Test queries with ground truth: {len(test_ground_truth):,}")
    
    # Evaluate Flat index
    print("\n" + "=" * 80)
    flat_eval, flat_avg_time, flat_total_time, flat_query_results = evaluate_index(
        flat_index, "FLAT", test_queries, test_ground_truth, 
        embedding_model, product_ids, k_values=[1, 5, 10, 20, 50, 100, 200]
    )
    print_evaluation_results(flat_eval, "FLAT", flat_avg_time, flat_total_time)
    
    # Evaluate HNSW index with increased ef_search
    print("\n" + "=" * 80)
    # Try different ef_search values for HNSW
    ef_search_values = [50, 100, 150, 200]  # Test multiple values
    
    hnsw_results_dict = {}
    for ef_val in ef_search_values:
        print(f"\n{'=' * 80}")
        print(f"Testing HNSW with ef_search={ef_val}")
        print(f"{'=' * 80}")
        hnsw_eval, hnsw_avg_time, hnsw_total_time, hnsw_query_results = evaluate_index(
            hnsw_index, "HNSW", test_queries, test_ground_truth,
            embedding_model, product_ids, k_values=[1, 5, 10, 20, 50, 100, 200],
            ef_search=ef_val
        )
        print_evaluation_results(hnsw_eval, f"HNSW (ef_search={ef_val})", hnsw_avg_time, hnsw_total_time)
        hnsw_results_dict[ef_val] = {
            'eval': hnsw_eval,
            'avg_time': hnsw_avg_time,
            'total_time': hnsw_total_time,
            'query_results': hnsw_query_results
        }
    
    # Use the best ef_search result for comparison (ef_search=150 as default)
    best_ef = 150
    hnsw_eval = hnsw_results_dict[best_ef]['eval']
    hnsw_avg_time = hnsw_results_dict[best_ef]['avg_time']
    hnsw_total_time = hnsw_results_dict[best_ef]['total_time']
    hnsw_query_results = hnsw_results_dict[best_ef]['query_results']
    
    # Compare indices
    compare_indices(flat_eval, hnsw_eval, flat_avg_time, hnsw_avg_time)
    
    # Compare different ef_search values for HNSW
    print("\n" + "=" * 80)
    print("HNSW EF_SEARCH COMPARISON")
    print("=" * 80)
    print(f"\n{'ef_search':<15} {'Recall@50':<15} {'Recall@100':<15} {'Recall@200':<15} {'Avg Time (ms)':<15}")
    print("-" * 80)
    for ef_val in sorted(hnsw_results_dict.keys()):
        eval_result = hnsw_results_dict[ef_val]['eval']
        avg_time = hnsw_results_dict[ef_val]['avg_time']
        recall_50 = eval_result.recall_at_k.get(50, 0.0)
        recall_100 = eval_result.recall_at_k.get(100, 0.0)
        recall_200 = eval_result.recall_at_k.get(200, 0.0)
        print(f"{ef_val:<15} {recall_50:<15.4f} {recall_100:<15.4f} {recall_200:<15.4f} {avg_time*1000:<15.2f}")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    results = {
        'flat': {
            'metrics': flat_eval.to_dict(),
            'avg_search_time_ms': flat_avg_time * 1000,
            'total_search_time_s': flat_total_time
        },
        'hnsw': {
            'metrics': hnsw_eval.to_dict(),
            'avg_search_time_ms': hnsw_avg_time * 1000,
            'total_search_time_s': hnsw_total_time,
            'ef_search': best_ef
        },
        'hnsw_all_ef_search': {
            ef_val: {
                'metrics': result['eval'].to_dict(),
                'avg_search_time_ms': result['avg_time'] * 1000,
                'total_search_time_s': result['total_time']
            }
            for ef_val, result in hnsw_results_dict.items()
        },
        'n_queries': len(test_queries),
        'n_queries_with_gt': len(test_ground_truth)
    }
    
    results_file = INDEX_DIR / "evaluation_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nQueries evaluated: {len(test_queries):,}")
    print(f"\nFlat Index:")
    print(f"  Recall@10: {flat_eval.recall_at_k[10]:.4f}")
    print(f"  Precision@10: {flat_eval.precision_at_k[10]:.4f}")
    print(f"  MRR: {flat_eval.mrr:.4f}")
    print(f"  Avg search time: {flat_avg_time*1000:.2f}ms")
    
    print(f"\nHNSW Index (ef_search={best_ef}):")
    print(f"  Recall@10: {hnsw_eval.recall_at_k[10]:.4f}")
    print(f"  Recall@50: {hnsw_eval.recall_at_k.get(50, 0.0):.4f}")
    print(f"  Recall@100: {hnsw_eval.recall_at_k.get(100, 0.0):.4f}")
    print(f"  Recall@200: {hnsw_eval.recall_at_k.get(200, 0.0):.4f}")
    print(f"  Precision@10: {hnsw_eval.precision_at_k[10]:.4f}")
    print(f"  MRR: {hnsw_eval.mrr:.4f}")
    print(f"  Avg search time: {hnsw_avg_time*1000:.2f}ms")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
