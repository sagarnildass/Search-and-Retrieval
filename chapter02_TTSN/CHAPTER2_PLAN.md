# Chapter 2: Two-Tower Architecture for Query-Product Matching

## Overview

This chapter implements a two-tower neural network architecture for semantic matching between queries and products. Unlike Chapter 1's single encoder, this approach uses separate encoders for queries and products, trained with contrastive learning to learn domain-specific representations.

## Objectives

- Implement two-tower architecture (query tower + product tower)
- Train with contrastive learning (triplet loss, in-batch negatives)
- Implement hard negative mining strategies
- Generate domain-specific embeddings
- Rebuild FAISS indices with trained embeddings
- Evaluate improvement over Chapter 1 baseline

## Key Concepts

### Two-Tower Architecture

```
Query Tower:  Query Text → Embedding (d-dim)
                    ↓
              Similarity Score
                    ↓
Product Tower: Product Text → Embedding (d-dim)
```

**Advantages:**
- Separate encoders can specialize for queries vs products
- Query embeddings computed on-the-fly (no pre-indexing needed)
- Product embeddings pre-computed and indexed
- Can fine-tune on domain-specific data

### Contrastive Learning

- **Goal**: Pull similar pairs together, push dissimilar pairs apart
- **Loss Functions**:
  - Triplet Loss: (anchor, positive, negative)
  - Contrastive Loss: (positive pair, negative pair)
  - In-batch Negatives: Use other queries' positives as negatives

## Dataset Preparation

### Input Data

- **Products**: `chapter01/data/products_clean.parquet` (1.8M products)
- **Examples**: `shopping_queries_dataset_examples.parquet` (2.6M query-product pairs)
- **Ground Truth**: `chapter01/ground_truth.pkl` (ESCI labels)

### Training Data Construction

#### 1. Positive Pairs

**Strategy**: Use ESCI labels to define positives with weights

```python
# Positive pairs with weights
- E (Exact):      weight = 1.0  (strong positive)
- S (Substitute): weight = 0.9  (positive)
- C (Complement): weight = 0.6  (weak positive, optional)
```

**Implementation:**
- Extract all (query, product_id) pairs where esci_label ∈ ['E', 'S', 'C']
- Store with relevance weights
- Expected: ~2.3M positive pairs

#### 2. Negative Pairs

**Strategy**: Multi-level negative sampling

**A. Random Negatives**
- Sample products not in query's relevant set
- Ratio: 1:2 to 1:4 (positive:negative)
- Easy negatives for initial training

**B. Hard Negatives**
- Use Chapter 1 FAISS system to find similar but irrelevant products
- Retrieve top-K candidates, filter out E/S/C products
- Keep I-labeled or not-in-ground-truth products
- Expected: 10-20 hard negatives per query

**C. In-Batch Negatives**
- Automatically use other queries' positives as negatives
- Efficient, no extra sampling needed
- Works well with contrastive loss

#### 3. Data Splits

```python
# Use existing split from dataset
train:  examples_df[examples_df['split'] == 'train']  # ~2.0M examples
val:    Sample 10% from train                          # ~200K examples  
test:   examples_df[examples_df['split'] == 'test']    # ~640K examples
```

#### 4. Data Format

```python
TrainingExample = {
    'query': str,
    'product_id': str,
    'product_text': str,  # From products_clean.parquet
    'label': float,       # 1.0 for positive, 0.0 for negative
    'weight': float,      # Relevance weight (1.0 for E, 0.9 for S, etc.)
    'neg_type': str,      # 'random', 'hard', 'in_batch'
    'split': str          # 'train', 'val', 'test'
}
```

## Implementation Plan

### Phase 1: Data Preparation

**Deliverable**: `chapter02_TTSN/prepare_training_data.py`

**Tasks:**
1. Load examples and products data
2. Create positive pairs from ESCI labels
3. Sample random negatives
4. Generate hard negatives using Chapter 1 FAISS system
5. Create train/val/test splits
6. Save training data in efficient format (parquet or PyTorch Dataset)

**Output:**
- `chapter02_TTSN/data/training_pairs.parquet`
- `chapter02_TTSN/data/val_pairs.parquet`
- `chapter02_TTSN/data/test_pairs.parquet`
- Statistics: positive/negative ratios, label distribution

### Phase 2: Two-Tower Model Architecture

**Deliverable**: `chapter02_TTSN/two_tower_model.py`

**Components:**

#### A. Query Tower
```python
class QueryTower(nn.Module):
    - Base encoder: SentenceTransformer (e.g., all-MiniLM-L6-v2)
    - Optional: Additional layers (MLP, normalization)
    - Output: Query embedding (d-dim)
```

#### B. Product Tower
```python
class ProductTower(nn.Module):
    - Base encoder: SentenceTransformer (same as query or separate)
    - Optional: Additional layers
    - Output: Product embedding (d-dim)
```

#### C. Two-Tower Model
```python
class TwoTowerModel(nn.Module):
    - query_tower: QueryTower
    - product_tower: ProductTower
    - similarity: Cosine similarity or dot product
    - forward(query, product) -> similarity_score
```

**Design Decisions:**
- **Shared vs Separate**: Start with shared base encoder, can experiment with separate
- **Embedding Dimension**: 384 (same as Chapter 1) or 512
- **Normalization**: L2 normalize embeddings for cosine similarity

### Phase 3: Loss Functions

**Deliverable**: `chapter02_TTSN/losses.py`

**Implement:**

#### A. Contrastive Loss
```python
def contrastive_loss(query_emb, product_emb, labels, margin=0.2):
    # Positive pairs: minimize distance
    # Negative pairs: maximize distance (with margin)
    # Weighted by relevance weights
```

#### B. Triplet Loss
```python
def triplet_loss(anchor, positive, negative, margin=0.2):
    # Pull anchor and positive together
    # Push anchor and negative apart
```

#### C. In-Batch Negative Loss
```python
def in_batch_negative_loss(query_embs, product_embs, labels):
    # Compute similarity matrix (batch_size x batch_size)
    # Positive diagonal, negatives off-diagonal
    # Cross-entropy loss
```

**Recommendation**: Start with weighted contrastive loss + in-batch negatives

### Phase 4: Training Script

**Deliverable**: `chapter02_TTSN/train_two_tower.py`

**Features:**
1. Load training data
2. Initialize two-tower model
3. Training loop with:
   - Batch sampling (with in-batch negatives)
   - Forward pass
   - Loss computation
   - Backward pass
   - Gradient clipping
   - Learning rate scheduling
4. Validation loop
5. Model checkpointing
6. Training metrics logging

**Training Configuration:**
- Batch size: 64-128 (larger = more in-batch negatives)
- Learning rate: 1e-5 to 5e-5
- Optimizer: AdamW
- Epochs: 3-5
- Gradient clipping: 1.0
- Warmup: 10% of steps

**Hard Negative Mining:**
- After epoch 1: Generate hard negatives using current model
- Retrain with hard negatives
- Iterate 2-3 times

### Phase 5: Hard Negative Mining

**Deliverable**: `chapter02_TTSN/hard_negative_mining.py`

**Strategy:**
1. Load current model (after initial training)
2. Generate product embeddings for all products
3. For each query:
   - Embed query
   - Retrieve top-K (e.g., 100) products
   - Filter out positives (E/S/C)
   - Keep I-labeled or not-in-ground-truth
   - Use top 10-20 as hard negatives
4. Add hard negatives to training data
5. Retrain model

**Iterative Process:**
- Epoch 1: Train with random negatives
- Epoch 2: Mine hard negatives, retrain
- Epoch 3: Mine harder negatives, fine-tune

### Phase 6: Inference and Reindexing

**Deliverable**: `chapter02_TTSN/reindex_with_trained_model.py`

**Tasks:**
1. Load trained two-tower model
2. Generate product embeddings using product tower
3. Rebuild FAISS indices (Flat + HNSW)
4. Save new indices to `chapter02_TTSN/data/indices/`
5. Update metadata with model information

**Note**: 
- Only reindex products (queries embedded on-the-fly)
- Keep Chapter 1 indices for comparison
- Save to separate directory

### Phase 7: Evaluation

**Deliverable**: `chapter02_TTSN/evaluate_trained_model.py`

**Tasks:**
1. Load trained model and new indices
2. Evaluate on test set
3. Compare with Chapter 1 baseline:
   - Recall@K, Precision@K, MRR, NDCG, MAP
   - Search latency
4. Generate comparison report

**Expected Improvements:**
- Recall@10: +10-20% improvement
- Recall@50: +15-25% improvement
- MRR: +0.05-0.15 improvement

### Phase 8: Demo Script

**Deliverable**: `chapter02_TTSN/demo_trained_model.py`

**Features:**
- Load trained two-tower model
- Load new FAISS indices
- Interactive search demo
- Compare with Chapter 1 baseline
- Show query and product embeddings

## File Structure

```
chapter02_TTSN/
├── CHAPTER2_PLAN.md              # This file
├── prepare_training_data.py      # Phase 1: Data preparation
├── two_tower_model.py            # Phase 2: Model architecture
├── losses.py                     # Phase 3: Loss functions
├── train_two_tower.py            # Phase 4: Training script
├── hard_negative_mining.py        # Phase 5: Hard negative mining
├── reindex_with_trained_model.py # Phase 6: Reindexing
├── evaluate_trained_model.py     # Phase 7: Evaluation
├── demo_trained_model.py         # Phase 8: Demo
├── utils.py                      # Utility functions
├── requirements.txt              # Additional dependencies
└── data/                         # Generated data
    ├── training_pairs.parquet
    ├── val_pairs.parquet
    ├── test_pairs.parquet
    ├── hard_negatives.parquet
    └── indices/                  # New FAISS indices
        ├── faiss_index_flat.bin
        ├── faiss_index_hnsw.bin
        ├── product_ids.pkl
        └── metadata.pkl
└── checkpoints/                  # Model checkpoints
    ├── epoch_1.pt
    ├── epoch_2.pt
    └── best_model.pt
```

## Training Strategy

### Stage 1: Initial Training (Epochs 1-2)
- **Data**: Positives (E+S) + Random negatives (1:2 ratio)
- **Loss**: Weighted contrastive loss + in-batch negatives
- **Goal**: Learn basic query-product matching

### Stage 2: Hard Negative Mining (Epoch 3)
- **Data**: Add hard negatives from Stage 1 model
- **Loss**: Same as Stage 1
- **Goal**: Improve discrimination on difficult cases

### Stage 3: Fine-tuning (Epochs 4-5)
- **Data**: Include C (Complement) as weak positives
- **Loss**: Weighted loss (E=1.0, S=0.9, C=0.6)
- **Learning Rate**: Reduce by 10x
- **Goal**: Refine model on edge cases

## Evaluation Metrics

### Retrieval Metrics (Same as Chapter 1)
- Recall@K (K=1, 5, 10, 20, 50, 100, 200)
- Precision@K
- MRR (Mean Reciprocal Rank)
- NDCG@K
- MAP (Mean Average Precision)

### Training Metrics
- Training loss (per epoch, per batch)
- Validation loss
- Learning rate schedule
- Gradient norms
- Embedding statistics

### Comparison Metrics
- Improvement over Chapter 1 baseline
- Speed comparison (embedding generation time)
- Memory usage

## Expected Outcomes

### Performance Targets

**Baseline (Chapter 1):**
- Recall@10: 11.56%
- Recall@50: 26.55%
- Recall@200: 40.78%
- MRR: 0.3686

**Target (Chapter 2):**
- Recall@10: 15-20% (+30-70% improvement)
- Recall@50: 35-45% (+30-70% improvement)
- Recall@200: 50-60% (+20-50% improvement)
- MRR: 0.45-0.55 (+20-50% improvement)

### Why These Targets Are Realistic

1. **Domain-specific training**: Model learns e-commerce specific patterns
2. **Hard negatives**: Better discrimination
3. **Separate towers**: Can specialize for queries vs products
4. **Contrastive learning**: Explicitly optimizes for similarity

## Implementation Timeline

### Week 1: Data Preparation & Model Architecture
- [ ] Day 1-2: Data preparation script
- [ ] Day 3-4: Two-tower model implementation
- [ ] Day 5: Loss functions implementation

### Week 2: Training
- [ ] Day 1-2: Training script
- [ ] Day 3-4: Initial training (Stage 1)
- [ ] Day 5: Hard negative mining implementation

### Week 3: Hard Negatives & Fine-tuning
- [ ] Day 1-2: Generate hard negatives, retrain (Stage 2)
- [ ] Day 3-4: Fine-tuning (Stage 3)
- [ ] Day 5: Model evaluation and tuning

### Week 4: Reindexing & Evaluation
- [ ] Day 1-2: Reindexing script and execution
- [ ] Day 3: Comprehensive evaluation
- [ ] Day 4-5: Demo script and documentation

## Dependencies

### Additional Packages (beyond Chapter 1)
```python
torch>=1.9.0              # Deep learning framework
transformers>=4.20.0      # Already have for QE
tensorboard>=2.10.0       # Training visualization (optional)
wandb>=0.12.0             # Experiment tracking (optional)
```

## Key Design Decisions

### 1. Shared vs Separate Encoders
- **Start with**: Shared base encoder (simpler, fewer parameters)
- **Can experiment with**: Separate encoders (more flexibility)

### 2. Embedding Dimension
- **Recommendation**: 384 (same as Chapter 1) for fair comparison
- **Alternative**: 512 (more capacity, slower)

### 3. Base Model
- **Start with**: `all-MiniLM-L6-v2` (same as Chapter 1)
- **Can upgrade**: `all-mpnet-base-v2` (better quality)

### 4. Loss Function
- **Primary**: Weighted contrastive loss
- **Secondary**: In-batch negatives
- **Optional**: Triplet loss for hard negatives

### 5. Hard Negative Strategy
- **Initial**: Use Chapter 1 FAISS system
- **Iterative**: Use current model after each epoch
- **Ratio**: 1:1 to 1:2 (positive:hard negative)

## Challenges and Solutions

### Challenge 1: Large Dataset
**Solution**: 
- Efficient data loading (PyTorch Dataset with caching)
- Batch processing
- Gradient accumulation if needed

### Challenge 2: Hard Negative Mining
**Solution**:
- Use FAISS for fast retrieval
- Batch embedding generation
- Cache hard negatives between epochs

### Challenge 3: Training Time
**Solution**:
- Use GPU if available
- Efficient batching
- Early stopping on validation loss
- Model checkpointing

### Challenge 4: Memory
**Solution**:
- Generate embeddings in batches
- Don't store all embeddings in memory
- Use gradient checkpointing if needed

## Success Criteria

### Must Have
- [ ] Two-tower model implemented and trained
- [ ] Recall@10 improvement of at least +20% over baseline
- [ ] New FAISS indices built and working
- [ ] Evaluation script showing improvements
- [ ] Demo script functional

### Nice to Have
- [ ] Training visualization (TensorBoard/W&B)
- [ ] Model interpretability (embedding analysis)
- [ ] A/B testing framework
- [ ] Production deployment considerations

## Next Steps After Chapter 2

### Chapter 3: Advanced Retrieval Techniques
- Optimize FAISS indices further
- Multi-GPU FAISS
- Index compression
- Batch query processing

### Chapter 4: Reranking
- Cross-encoder reranking (already have in Chapter 1)
- Learning-to-rank
- Feature engineering

## Notes

- Keep Chapter 1 indices for baseline comparison
- Save model checkpoints frequently
- Log all hyperparameters and results
- Document any deviations from plan
- Test on validation set before final evaluation

## References

- [Two-Tower Architecture Paper](https://research.google/pubs/pub48840/)
- [Contrastive Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/)
- [Hard Negative Mining](https://arxiv.org/abs/1704.04920)
- [PyTorch Training Tutorial](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
