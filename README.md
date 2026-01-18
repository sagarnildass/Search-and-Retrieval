# Search Ads Retrieval: From Zero to Advanced

## Course Overview

This course provides a hands-on journey from basic vector retrieval to advanced semantic ID-based multiphase retrieval systems, culminating in implementing the MMQ (Multimodal Mixture-of-Quantization) framework for search ads.

## Prerequisites

- Python programming (intermediate level)
- Basic understanding of machine learning
- Familiarity with PyTorch/TensorFlow (helpful but not required)

---

## Chapter 1: Foundations - Basic Vector Retrieval with FAISS

Objective: Build a working retrieval system from scratch using FAISS

Topics:

- Introduction to search ads and retrieval systems
- Understanding embeddings and vector spaces
- FAISS basics: IndexFlatL2, IndexIVFFlat, IndexHNSW
- Building a product catalog index
- Query processing and similarity search
- Evaluation metrics: Recall@K, Precision@K, MRR

Hands-on Exercises:

1.  Generate embeddings for product descriptions using sentence transformers
2.  Build and query a FAISS index
3.  Implement recall evaluation
4.  Compare exact vs. approximate search performance

Dataset: Amazon Product Dataset (Small Subset)

- Source: [Amazon Product Data](https://nijianmo.github.io/amazon/index.html)
- Use: "Small" subset (1-2M products) with metadata
- Fields: product title, description, category, price
- Availability: Public, downloadable
- Why: Real-world e-commerce data, manageable size for learning

Deliverables:

- `chapter1_basic_retrieval.py` - Complete retrieval pipeline
- `chapter1_evaluation.py` - Evaluation metrics implementation

---

## Chapter 2: Two-Tower Architecture for Query-Product Matching

Objective: Implement a two-tower neural network for semantic matching

Topics:

- Two-tower architecture: query tower vs. product tower
- Training with contrastive learning (triplet loss, in-batch negatives)
- Hard negative mining strategies
- Embedding normalization and similarity computation
- Integration with FAISS for efficient retrieval

Hands-on Exercises:

1.  Build query and product encoder towers
2.  Train with contrastive loss
3.  Generate embeddings for retrieval
4.  Compare two-tower vs. single-encoder performance

Dataset: Amazon Product Dataset + Query Logs

- Use: Product data + synthetic/real user queries
- Create query-product pairs from:
- User reviews (treat review text as queries)
- Search logs if available
- Synthetic queries from product titles
- Availability: Extend Chapter 1 dataset
- Why: Need query-product pairs for training

Deliverables:

- `chapter2_two_tower.py` - Two-tower model implementation
- `chapter2_training.py` - Training script with contrastive loss
- `chapter2_inference.py` - Inference and FAISS integration

---

## Chapter 3: Advanced Retrieval Techniques - ANN Optimization

Objective: Optimize retrieval for scale and speed

Topics:

- FAISS index types: IVF, HNSW, PQ (Product Quantization)
- Index parameter tuning (nlist, m, efConstruction, efSearch)
- Multi-GPU FAISS
- Batch query processing
- Recall-latency tradeoffs
- Index compression techniques

Hands-on Exercises:

1.  Benchmark different FAISS index types
2.  Tune HNSW parameters for optimal recall-latency
3.  Implement batch retrieval
4.  Build a retrieval latency dashboard

Dataset: Amazon Product Dataset (Full/Large Subset)

- Use: 5-10M products to demonstrate scalability
- Availability: Public, larger subset
- Why: Need scale to see ANN benefits

Deliverables:

- `chapter3_ann_benchmark.py` - Index comparison tool
- `chapter3_optimization.py` - Parameter tuning utilities

---

## Chapter 4: Reranking and Two-Stage Retrieval

Objective: Implement reranking to improve precision after initial retrieval

Topics:

- Two-stage retrieval architecture
- Cross-encoder rerankers (BERT-based)
- Learning-to-rank approaches
- Feature engineering for reranking
- End-to-end pipeline: FAISS retrieval → Reranking

Hands-on Exercises:

1.  Build a cross-encoder reranker
2.  Implement feature-based reranking
3.  Create two-stage pipeline
4.  Evaluate recall-precision tradeoffs

Dataset: Amazon Product Dataset + Relevance Labels

- Create relevance labels:
- Click-through data (if available)
- Manual annotations on query-product pairs
- Synthetic labels based on category match
- Availability: Extend previous datasets
- Why: Need relevance labels for reranking training

Deliverables:

- `chapter4_reranker.py` - Cross-encoder reranker
- `chapter4_two_stage.py` - Complete two-stage pipeline

---

## Chapter 5: Semantic IDs with RQVAE (Residual Quantized VAE)

Objective: Generate hierarchical semantic IDs for products and queries

Topics:

- Introduction to vector quantization and semantic IDs
- RQVAE architecture: residual quantization layers
- Training RQVAE for product embeddings
- Generating semantic ID sequences
- Hierarchical semantic understanding (coarse → fine)

Hands-on Exercises:

1.  Implement RQVAE from scratch
2.  Train on product embeddings
3.  Generate semantic IDs for products
4.  Visualize semantic ID hierarchy

Dataset: Amazon Product Dataset (with embeddings from Chapter 2)

- Use: Product embeddings from two-tower model
- Train RQVAE on these embeddings
- Availability: Use Chapter 2 outputs
- Why: Semantic IDs work on embeddings, not raw data

Deliverables:

- `chapter5_rqvae.py` - RQVAE implementation
- `chapter5_semantic_ids.py` - Semantic ID generation
- `chapter5_visualization.py` - ID hierarchy visualization

---

## Chapter 6: Multiphase Retrieval System

Objective: Build a hierarchical retrieval system using semantic IDs

Topics:

- Multiphase retrieval architecture
- Phase 1: Semantic ID matching (coarse filtering)
- Phase 2: Fine-grained vector search (FAISS)
- Hierarchical semantic matching strategies
- Combining multiple retrieval phases
- Performance optimization

Hands-on Exercises:

1.  Build semantic ID index (first phase)
2.  Implement hierarchical matching (coarse → fine)
3.  Integrate with FAISS for second phase
4.  Evaluate multiphase vs. single-phase retrieval

Dataset: Amazon Product Dataset (Full)

- Use: Products with semantic IDs from Chapter 5
- Create query semantic IDs
- Availability: Use Chapter 5 outputs
- Why: Need semantic IDs for hierarchical matching

Deliverables:

- `chapter6_multiphase.py` - Multiphase retrieval system
- `chapter6_evaluation.py` - Comparison with baseline

---

## Chapter 7: Multimodal Retrieval - Text + Image

Objective: Extend retrieval to handle multimodal product data

Topics:

- Multimodal embeddings (CLIP, ALIGN)
- Combining text and image features
- Multimodal FAISS indexing
- Query understanding: text-only vs. image queries
- Cross-modal retrieval

Hands-on Exercises:

1.  Extract image features using vision encoders
2.  Combine text and image embeddings
3.  Build multimodal FAISS index
4.  Implement cross-modal retrieval

Dataset: Amazon Product Dataset with Images

- Source: Amazon Product Dataset (image URLs available)
- Alternative: Fashion-MNIST or eBay Product Dataset
- Fields: Product images + text descriptions
- Availability: Public (images may need downloading)
- Why: Need both text and images for multimodal learning

Deliverables:

- `chapter7_multimodal.py` - Multimodal embedding extraction
- `chapter7_indexing.py` - Multimodal FAISS index
- `chapter7_retrieval.py` - Cross-modal retrieval

---

## Chapter 8: MMQ Implementation - Multimodal Mixture-of-Quantization

Objective: Recreate the MMQ framework from the paper

Topics:

- MMQ architecture: multi-expert system
- Modality-specific vs. modality-shared experts
- Orthogonal regularization for expert diversity
- Two-stage training: shared-specific tokenizer + behavior-aware fine-tuning
- Multimodal reconstruction loss
- Semantic ID generation with MMQ

Hands-on Exercises:

1.  Implement multi-expert architecture
2.  Train shared-specific tokenizer with orthogonal regularization
3.  Implement behavior-aware fine-tuning
4.  Generate MMQ semantic IDs
5.  Evaluate on retrieval tasks

Dataset: Multimodal Product Dataset (Text + Images)

- Use: Same as Chapter 7 (Amazon with images)
- Need: User behavior data (clicks, purchases) if available
- Alternative: Synthetic behavior from product interactions
- Availability: Public (may need to create behavior labels)
- Why: MMQ requires multimodal data + behavior signals

Deliverables:

- `chapter8_mmq_architecture.py` - Multi-expert architecture
- `chapter8_tokenizer.py` - Shared-specific tokenizer
- `chapter8_finetuning.py` - Behavior-aware fine-tuning
- `chapter8_evaluation.py` - Full MMQ evaluation

---

## Chapter 9: Production Deployment and Optimization

Objective: Prepare retrieval system for production use

Topics:

- Serving FAISS indices at scale
- Real-time query processing
- Caching strategies
- Monitoring and observability
- A/B testing retrieval systems
- Performance optimization

Hands-on Exercises:

1.  Build a REST API for retrieval
2.  Implement caching layer
3.  Add monitoring and logging
4.  Create A/B testing framework

Dataset: Use all previous datasets

- Why: Production concerns apply to all systems

Deliverables:

- `chapter9_api.py` - REST API server
- `chapter9_monitoring.py` - Monitoring utilities
- `chapter9_ab_testing.py` - A/B testing framework

---

## Chapter 10: Capstone Project - End-to-End Search Ads System

Objective: Build a complete search ads retrieval and ranking system

Project Components:

1.  Multiphase retrieval with semantic IDs
2.  MMQ-based semantic ID generation
3.  Two-stage ranking (retrieval + reranking)
4.  Production-ready API
5.  Comprehensive evaluation

Dataset: Complete Amazon Product Dataset + Custom Queries

- Use: Full dataset with all features
- Create: Test query set with ground truth
- Why: Real-world end-to-end evaluation

Deliverables:

- Complete system implementation
- Evaluation report
- Performance benchmarks

---

## Dataset Summary

| Chapter | Dataset | Availability | Size |\
| ------- | ------------------------------ | ----------------- | -------------- |\
| 1-2 | Amazon Product Dataset (Small) | Public | 1-2M products |\
| 3 | Amazon Product Dataset (Large) | Public | 5-10M products |\
| 4 | Amazon + Relevance Labels | Create labels | Same as Ch 1-2 |\
| 5 | Embeddings from Ch 2 | Generated | Same as Ch 2 |\
| 6 | Semantic IDs from Ch 5 | Generated | Same as Ch 5 |\
| 7-8 | Amazon with Images | Public (download) | 1-5M products |\
| 9-10 | All previous | Combined | Full scale |

---

## Course Structure

- Format: Jupyter notebooks + Python scripts
- Progression: Each chapter builds on previous
- Hands-on: Every chapter has working code
- Evaluation: Metrics and benchmarks throughout
- Final Project: Complete system implementation

---

## Technical Stack

- Vector DB: FAISS (Facebook AI Similarity Search)
- Deep Learning: PyTorch
- Embeddings: Sentence Transformers, CLIP
- Evaluation: Custom metrics + standard IR metrics
- Deployment: FastAPI, Docker (optional)

---

## Learning Outcomes

By the end of this course, you will:

1.  Understand vector retrieval systems from basics to advanced
2.  Implement two-tower architectures for semantic matching
3.  Build multiphase retrieval systems with semantic IDs
4.  Recreate state-of-the-art MMQ framework
5.  Deploy production-ready retrieval systems
6.  Optimize for scale, speed, and accuracy
