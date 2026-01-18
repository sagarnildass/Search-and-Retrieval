# Chapter 1: Foundations - Basic Vector Retrieval with FAISS

## Overview

This chapter implements a complete vector retrieval system from scratch using FAISS, including data preparation, index building, evaluation, and advanced features like query expansion and cross-encoder reranking.

## Objectives

- Build working retrieval systems using FAISS (Flat and HNSW indices)
- Generate embeddings using sentence transformers
- Evaluate retrieval performance with standard IR metrics
- Implement advanced techniques: query expansion and reranking
- Handle large-scale datasets (1.8M+ products) efficiently

## Dataset

**Shopping Queries Dataset**
- Products: 1,814,924 products with titles, descriptions, bullet points, brand, color
- Queries: 130,193 unique queries with relevance labels (ESCI)
- Ground Truth: E (Exact), S (Substitute), C (Complement) = relevant; I (Irrelevant) = not relevant
- Average relevant products per query: 18.0

## File Structure

```
chapter01/
├── README.md                          # This file
├── 01_eda_and_data_prep.ipynb        # Data exploration and preparation notebook
├── build_index.py                     # Script to build FAISS indices
├── evaluate_indices.py                # Evaluation script with metrics
├── demo_retrieval.py                  # Basic retrieval demo
├── demo_retrieval_expansion_rerank.py # Advanced demo with QE + reranking
├── requirements.txt                   # Python dependencies
├── ground_truth.pkl                   # Ground truth labels for evaluation
└── data/                              # Generated data directory
    ├── faiss_index_flat.bin          # Flat (exact) FAISS index
    ├── faiss_index_hnsw.bin          # HNSW (approximate) FAISS index
    ├── product_ids.pkl                # Product ID mapping
    ├── products_clean.parquet         # Cleaned product data
    ├── metadata.pkl                   # Index metadata
    └── evaluation_results.pkl         # Evaluation results
```

## Installation

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### Required Packages

- `numpy>=1.21.0`
- `faiss-cpu>=1.7.4` (or `faiss-gpu` if you have CUDA)
- `sentence-transformers>=2.2.0`
- `torch>=1.9.0`
- `transformers>=4.20.0` (for query expansion)
- `pandas>=1.3.0`
- `pyarrow>=5.0.0` (for parquet support)
- `tqdm>=4.60.0`

## Usage Guide

### 1. Data Preparation

**Notebook: `01_eda_and_data_prep.ipynb`**

This notebook:
- Loads and explores the dataset
- Cleans and prepares product text
- Creates ground truth labels
- Generates embeddings
- Builds FAISS indices
- Saves everything to disk

**Run cells sequentially** to process the data. The notebook handles:
- Memory-efficient processing (chunked)
- Progress tracking
- Error handling

### 2. Build Indices (Alternative to Notebook)

**Script: `build_index.py`**

If you prefer a script over a notebook:

```bash
cd chapter01
python build_index.py
```

This script:
- Loads and cleans product data
- Generates embeddings in chunks (memory-efficient)
- Builds Flat and HNSW indices incrementally
- Saves all artifacts to `data/` directory

**Key Features:**
- Processes 100K products per chunk
- Handles 1.8M+ products without memory issues
- Builds both exact (Flat) and approximate (HNSW) indices
- Saves metadata for later use

### 3. Evaluate Indices

**Script: `evaluate_indices.py`**

Evaluates retrieval performance:

```bash
python evaluate_indices.py
```

**What it does:**
- Loads indices and ground truth
- Evaluates Flat and HNSW indices
- Tests multiple `ef_search` values for HNSW
- Calculates metrics: Recall@K, Precision@K, MRR, NDCG@K, MAP
- Compares performance
- Saves results to `data/evaluation_results.pkl`

**Configuration:**
- Set `sample_size` in the script to evaluate on a subset (default: 1000 queries)
- Set to `None` to evaluate on all queries (takes longer)

**Example Output:**
```
Flat Index:
  Recall@10: 0.1190
  Recall@50: 0.2732
  Recall@200: 0.4187
  Avg search time: 184.34ms

HNSW Index (ef_search=200):
  Recall@10: 0.1156
  Recall@50: 0.2655
  Recall@200: 0.4078
  Avg search time: 2.28ms (80x faster!)
```

### 4. Basic Retrieval Demo

**Script: `demo_retrieval.py`**

Simple interactive retrieval demo:

```bash
python demo_retrieval.py
```

**Features:**
- Search with Flat or HNSW index
- Compare both indices
- Interactive mode for custom queries
- Uses optimal settings (ef_search=200, K=20)

### 5. Advanced Retrieval Demo (Query Expansion + Reranking)

**Script: `demo_retrieval_expansion_rerank.py`**

Advanced demo with query expansion and reranking:

```bash
python demo_retrieval_expansion_rerank.py
```

**Features:**
- **Query Expansion**: Uses Flan-T5 to generate query variants
- **Cross-Encoder Reranking**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank results
- **Configurable**: Toggle QE and reranking on/off
- **Detailed Timing**: Shows breakdown of each step
- **Interactive**: Test different configurations

**Configuration Options:**
- Toggle query expansion on/off
- Toggle reranking on/off
- Adjust HNSW `ef_search` parameter
- Custom K values

**Example Usage:**
```python
from demo_retrieval_expansion_rerank import RetrievalDemoQE_Rerank, SearchConfig, ExpansionConfig, RerankConfig

# Customize configs
search_cfg = SearchConfig(default_k=20, candidate_k=100)
exp_cfg = ExpansionConfig(enabled=True, num_expansions=4)
rerank_cfg = RerankConfig(enabled=True, top_n=50)

demo = RetrievalDemoQE_Rerank(
    search_cfg=search_cfg,
    exp_cfg=exp_cfg,
    rerank_cfg=rerank_cfg
)
demo.load()

# Search
result = demo.search("wireless headphones", k=10)
demo.display_results(result)
```

## Performance Results

### Baseline Performance (without QE/Reranking)

**Flat Index (Exact Search):**
- Recall@10: 11.90%
- Recall@50: 27.32%
- Recall@200: 41.87%
- Average search time: 184.34ms

**HNSW Index (Approximate Search, ef_search=200):**
- Recall@10: 11.56%
- Recall@50: 26.55%
- Recall@200: 40.78%
- Average search time: 2.28ms (80x faster than Flat)
- MRR: 0.3686
- MAP: 0.2804

### Optimal Settings

- **Index Type**: HNSW
- **ef_search**: 200 (best recall/speed trade-off)
- **K for retrieval**: 100-200 (for better recall)
- **K for final results**: 20 (good balance)

### Expected Improvements with Advanced Features

- **Query Expansion**: +5-10% recall improvement
- **Cross-Encoder Reranking**: +15-30% precision improvement
- **Combined**: Better overall retrieval quality

## Key Components

### 1. Embedding Model

**Model**: `all-MiniLM-L6-v2`
- Dimensions: 384
- Fast and efficient
- Good baseline for retrieval

**Note**: Can be upgraded to `all-mpnet-base-v2` (768 dim) or `all-MiniLM-L12-v2` (384 dim) for better quality.

### 2. FAISS Indices

**Flat Index (IndexFlatL2)**
- Exact search
- Best recall
- Slower for large datasets

**HNSW Index (IndexHNSWFlat)**
- Approximate search
- Fast (2-3ms per query)
- Slightly lower recall than Flat
- Parameters:
  - `m=32`: Number of bi-directional links
  - `ef_construction=200`: Construction parameter
  - `ef_search=200`: Search parameter (optimal)

### 3. Query Expansion

**Model**: `google/flan-t5-base`
- Open-source instruction-tuned model
- Generates query variants/synonyms
- Improves recall for short/ambiguous queries

**Alternative**: Can be swapped for LLaMA or other models.

### 4. Cross-Encoder Reranking

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Pre-trained on MS MARCO dataset
- Scores query-product pairs
- Significantly improves precision

## Evaluation Metrics

### Recall@K
Fraction of relevant items retrieved in top K results.
- Higher is better
- Measures coverage of relevant items

### Precision@K
Fraction of retrieved items that are relevant.
- Higher is better
- Measures quality of results

### MRR (Mean Reciprocal Rank)
Average reciprocal rank of the first relevant item.
- Higher is better (0-1)
- Measures how quickly relevant items appear

### NDCG@K (Normalized Discounted Cumulative Gain)
Measures ranking quality considering position.
- Higher is better (0-1)
- Accounts for relevance at different positions

### MAP (Mean Average Precision)
Average precision across all queries.
- Higher is better (0-1)
- Overall ranking quality metric

## Troubleshooting

### Memory Issues

If you encounter memory errors:
1. Reduce chunk size in `build_index.py` (currently 100K)
2. Reduce batch size for embeddings (currently 4)
3. Process in smaller subsets

### Slow Query Expansion

Query expansion can be slow on CPU:
1. Use GPU if available (automatically detected)
2. Reduce `num_expansions` in config
3. Disable QE if not needed

### Slow Reranking

Reranking adds latency:
1. Reduce `top_n` in RerankConfig (fewer candidates to rerank)
2. Use smaller batch size
3. Consider GPU acceleration

## Next Steps

### Immediate Improvements

1. **Better Embedding Model**: Rebuild with `all-mpnet-base-v2` for +10-20% recall
2. **Hybrid Search**: Add BM25 for keyword matching
3. **Fine-tune Reranker**: Train on your specific data

### Chapter 2: Two-Tower Architecture

The next chapter will implement:
- Separate query and product encoders
- Contrastive learning (triplet loss)
- Hard negative mining
- Domain-specific fine-tuning

This should significantly improve recall (expected +20-30%).

## References

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)
- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)

## Summary

Chapter 1 successfully implements:
✅ Complete data pipeline for 1.8M+ products
✅ FAISS indices (Flat and HNSW) with optimal settings
✅ Comprehensive evaluation framework
✅ Query expansion with open-source models
✅ Cross-encoder reranking
✅ Production-ready code with error handling
✅ Interactive demos for testing

**Performance**: ~41% recall@200 with HNSW at 2.3ms per query - a solid foundation for building more advanced retrieval systems!
