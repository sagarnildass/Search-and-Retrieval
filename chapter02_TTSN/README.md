# Chapter 2: Two-Tower Semantic Network (TTSN)

## Overview

This chapter implements a two-tower neural network architecture for semantic matching between queries and products, trained with contrastive learning.

## Files

- `prepare_training_data.py` - Prepares training data with stage markers
- `two_tower_model.py` - Two-tower model architecture
- `losses.py` - Loss functions (contrastive, InfoNCE, combined)
- `train_two_tower.py` - Training script with stage management
- `hard_negative_mining.py` - Hard negative mining (to be implemented)
- `reindex_with_trained_model.py` - Reindexing with trained model (to be implemented)
- `evaluate_trained_model.py` - Evaluation script (to be implemented)

## Setup

1. Install dependencies:
```bash
pip install torch transformers sentence-transformers pandas numpy faiss-cpu tqdm
```

2. Prepare training data (already done):
```bash
python prepare_training_data.py
```

## Training

### Train All Stages (Recommended)

```bash
python train_two_tower.py \
    --data_dir data \
    --checkpoint_dir checkpoints \
    --batch_size 64 \
    --num_workers 4
```

### Train Specific Stage

```bash
# Stage 1 only
python train_two_tower.py --stage 1

# Stage 2 only (after Stage 1 and hard negative mining)
python train_two_tower.py --stage 2

# Stage 3 only (after Stage 2)
python train_two_tower.py --stage 3
```

### Training Stages

**Stage 1: Initial Training (Epochs 1-2)**
- Data: E+S positives + Random negatives
- Learning Rate: 2e-5
- Goal: Learn basic query-product matching

**Stage 2: Hard Negative Training (Epoch 3)**
- Data: E+S positives + Random + Hard negatives
- Learning Rate: 2e-5
- Goal: Improve discrimination on difficult cases
- **Note**: Requires hard negatives from Stage 1 model

**Stage 3: Fine-tuning (Epochs 4-5)**
- Data: E+S+C positives + Random + Hard negatives
- Learning Rate: 2e-6 (10x reduction)
- Goal: Refine model on edge cases

## Training Workflow

1. **Stage 1**: Train epochs 1-2
   ```bash
   python train_two_tower.py --stage 1
   ```

2. **Mine Hard Negatives** (after Stage 1):
   ```bash
   python hard_negative_mining.py \
       --checkpoint checkpoints/epoch_2.pt \
       --output data/hard_negatives_stage2.parquet
   ```
   (Note: This script needs to be implemented)

3. **Stage 2**: Train epoch 3 with hard negatives
   ```bash
   python train_two_tower.py --stage 2
   ```

4. **Stage 3**: Fine-tune epochs 4-5
   ```bash
   python train_two_tower.py --stage 3
   ```

## Checkpoints

Checkpoints are saved in `checkpoints/`:
- `epoch_1.pt`, `epoch_2.pt`, etc. - Per-epoch checkpoints
- `best_model.pt` - Best model based on validation loss

## Model Architecture

- **Query Tower**: Encodes queries using SentenceTransformer
- **Product Tower**: Encodes products using SentenceTransformer (shared encoder)
- **Similarity**: Cosine similarity between query and product embeddings
- **Embedding Dimension**: 384 (same as Chapter 1)

## Loss Functions

- **Weighted Contrastive Loss**: Pulls positives together, pushes negatives apart
- **In-Batch Negative Loss (InfoNCE)**: Uses other batch items as negatives
- **Combined Loss**: Weighted combination of both losses

## Expected Results

- **Baseline (Chapter 1)**: Recall@10 = 11.56%, MRR = 0.3686
- **Target (Chapter 2)**: Recall@10 = 15-20%, MRR = 0.45-0.55
