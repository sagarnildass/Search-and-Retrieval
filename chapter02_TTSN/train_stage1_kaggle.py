"""
Standalone Script for Stage 1 Training on Kaggle

This script contains all necessary code for training Stage 1 of the Two-Tower Model.
Includes: Model architecture, loss functions, and training loop.

Data paths (Kaggle):
- /kaggle/input/search-and-retrieval-ttsn-training-data/training_pairs.parquet
- /kaggle/input/search-and-retrieval-ttsn-training-data/val_pairs.parquet
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from typing import Dict, Optional, Tuple
import time
import shutil

# Set environment variables to avoid tokenizer warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class QueryTower(nn.Module):
    """Query encoder tower"""
    
    def __init__(
        self,
        base_model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        normalize: bool = True
    ):
        super(QueryTower, self).__init__()
        
        from sentence_transformers import SentenceTransformer
        self.base_encoder = SentenceTransformer(base_model_name)
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        
    def forward(self, query_texts):
        """Encode queries into embeddings with gradient support"""
        if isinstance(query_texts, str):
            query_texts = [query_texts]
        
        # Get device
        try:
            device = next(self.parameters()).device
        except:
            device = torch.device('cpu')
        
        # Access underlying model and tokenizer for gradient support
        # SentenceTransformer structure: [0] is the transformer model
        model = self.base_encoder[0].auto_model
        tokenizer = self.base_encoder.tokenizer
        
        # Tokenize
        encoded = tokenizer(
            query_texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        
        # Move to device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Forward pass through model
        model_output = model(**encoded)
        
        # Apply mean pooling (SentenceTransformer default)
        # Get token embeddings
        token_embeddings = model_output[0]  # [batch_size, seq_len, hidden_dim]
        attention_mask = encoded['attention_mask']  # [batch_size, seq_len]
        
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask  # [batch_size, hidden_dim]
        
        # Normalize
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class ProductTower(nn.Module):
    """Product encoder tower"""
    
    def __init__(
        self,
        base_model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        normalize: bool = True,
        shared_encoder: Optional[nn.Module] = None
    ):
        super(ProductTower, self).__init__()
        
        if shared_encoder is not None:
            self.base_encoder = shared_encoder
        else:
            from sentence_transformers import SentenceTransformer
            self.base_encoder = SentenceTransformer(base_model_name)
        
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        
    def forward(self, product_texts):
        """Encode products into embeddings with gradient support"""
        if isinstance(product_texts, str):
            product_texts = [product_texts]
        
        # Get device
        try:
            device = next(self.parameters()).device
        except:
            device = torch.device('cpu')
        
        # Access underlying model and tokenizer for gradient support
        # SentenceTransformer structure: [0] is the transformer model
        model = self.base_encoder[0].auto_model
        tokenizer = self.base_encoder.tokenizer
        
        # Tokenize
        encoded = tokenizer(
            product_texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        
        # Move to device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Forward pass through model
        model_output = model(**encoded)
        
        # Apply mean pooling (SentenceTransformer default)
        # Get token embeddings
        token_embeddings = model_output[0]  # [batch_size, seq_len, hidden_dim]
        attention_mask = encoded['attention_mask']  # [batch_size, seq_len]
        
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask  # [batch_size, hidden_dim]
        
        # Normalize
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class TwoTowerModel(nn.Module):
    """Two-tower model for query-product matching"""
    
    def __init__(
        self,
        base_model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        shared_encoder: bool = True,
        normalize: bool = True
    ):
        super(TwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        
        # Create query tower
        self.query_tower = QueryTower(
            base_model_name=base_model_name,
            embedding_dim=embedding_dim,
            normalize=normalize
        )
        
        # Create product tower (shared encoder)
        if shared_encoder:
            self.product_tower = ProductTower(
                base_model_name=base_model_name,
                embedding_dim=embedding_dim,
                normalize=normalize,
                shared_encoder=self.query_tower.base_encoder
            )
        else:
            self.product_tower = ProductTower(
                base_model_name=base_model_name,
                embedding_dim=embedding_dim,
                normalize=normalize,
                shared_encoder=None
            )
    
    def encode_queries(self, query_texts):
        """Encode queries only"""
        return self.query_tower(query_texts)
    
    def encode_products(self, product_texts):
        """Encode products only"""
        return self.product_tower(product_texts)
    
    def forward(self, query_texts, product_texts, return_embeddings=False):
        """Forward pass"""
        query_embs = self.query_tower(query_texts)
        product_embs = self.product_tower(product_texts)
        similarity = torch.sum(query_embs * product_embs, dim=1)
        
        if return_embeddings:
            return similarity, query_embs, product_embs
        else:
            return similarity


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def weighted_contrastive_loss(
    query_embs: torch.Tensor,
    product_embs: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    margin: float = 0.2
) -> torch.Tensor:
    """Weighted contrastive loss"""
    distances = torch.sum((query_embs - product_embs) ** 2, dim=1)
    
    positive_mask = (labels == 1.0).float()
    positive_loss = (weights * distances * positive_mask).sum()
    n_positives = positive_mask.sum()
    
    negative_mask = (labels == 0.0).float()
    negative_distances = distances * negative_mask
    negative_loss = (torch.clamp(margin - negative_distances, min=0.0) ** 2).sum()
    n_negatives = negative_mask.sum()
    
    if n_positives > 0 and n_negatives > 0:
        total_loss = (positive_loss / n_positives) + (negative_loss / n_negatives)
    elif n_positives > 0:
        total_loss = positive_loss / n_positives
    elif n_negatives > 0:
        total_loss = negative_loss / n_negatives
    else:
        total_loss = torch.tensor(0.0, device=query_embs.device)
    
    return total_loss


def in_batch_negative_loss(
    query_embs: torch.Tensor,
    product_embs: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.05
) -> torch.Tensor:
    """In-batch negative loss (InfoNCE)"""
    similarity_matrix = torch.matmul(query_embs, product_embs.T)  # [B, B]
    similarity_matrix = similarity_matrix / temperature
    
    positive_scores = torch.diag(similarity_matrix)  # [B]
    log_probs = positive_scores - torch.logsumexp(similarity_matrix, dim=1)
    
    positive_mask = (labels == 1.0).float()
    loss = -(log_probs * positive_mask).sum()
    
    n_positives = positive_mask.sum()
    if n_positives > 0:
        loss = loss / n_positives
    else:
        loss = torch.tensor(0.0, device=query_embs.device)
    
    return loss


def combined_loss(
    query_embs: torch.Tensor,
    product_embs: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    contrastive_weight: float = 0.5,
    inbatch_weight: float = 0.5,
    margin: float = 0.2,
    temperature: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined loss: weighted contrastive + in-batch negatives"""
    contrastive = weighted_contrastive_loss(
        query_embs, product_embs, labels, weights, margin
    )
    inbatch = in_batch_negative_loss(
        query_embs, product_embs, labels, temperature
    )
    
    total_loss = contrastive_weight * contrastive + inbatch_weight * inbatch
    
    return total_loss, contrastive, inbatch


# ============================================================================
# DATASET
# ============================================================================

class TwoTowerDataset(Dataset):
    """Dataset that filters by training stage"""
    
    def __init__(self, df: pd.DataFrame, stage: int = 1):
        self.df = df.copy()
        self.stage = stage
        
        # Filter data based on stage
        stage_column = f'use_in_stage_{stage}'
        if stage_column not in self.df.columns:
            raise ValueError(f"DataFrame must have '{stage_column}' column")
        
        self.df = self.df[self.df[stage_column] == True].copy()
        
        print(f"\nStage {stage} dataset:")
        print(f"  Total examples: {len(self.df):,}")
        print(f"  Positives: {(self.df['label'] == 1.0).sum():,}")
        print(f"  Negatives: {(self.df['label'] == 0.0).sum():,}")
        print(f"  Random negatives: {(self.df['neg_type'] == 'random').sum():,}")
        print(f"  Hard negatives: {(self.df['neg_type'] == 'hard').sum():,}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'query': str(row['query']),
            'product_text': str(row['product_text']),
            'label': torch.tensor(row['label'], dtype=torch.float32),
            'weight': torch.tensor(row['weight'], dtype=torch.float32)
        }


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def validate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Validating", leave=False):
            queries = batch['query']
            products = batch['product_text']
            labels = batch['label'].to(device)
            weights = batch['weight'].to(device)
            
            query_embs = model.encode_queries(queries)
            product_embs = model.encode_products(products)
            
            loss, _, _ = combined_loss(query_embs, product_embs, labels, weights)
            
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return avg_loss


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    steps_completed: int = 0,
    checkpoint_dir: Optional[Path] = None,
    start_batch: int = 0
) -> Tuple[Dict, int]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_contrastive = 0.0
    total_inbatch = 0.0
    n_batches = 0
    
    # Moving average for smoother loss tracking (window of 50 batches)
    loss_window = []
    contrastive_window = []
    inbatch_window = []
    window_size = 50
    
    # Skip batches if resuming mid-epoch
    if start_batch > 0:
        print(f"  Resuming from batch {start_batch}/{len(train_loader)}")
        # Create iterator and skip batches
        train_iter = iter(train_loader)
        for _ in range(start_batch):
            next(train_iter, None)
        batch_iterator = enumerate(train_iter, start=start_batch)
    else:
        batch_iterator = enumerate(tqdm(train_loader, desc=f"  Epoch {epoch}"))
    
    for batch_idx, batch in batch_iterator:
        queries = batch['query']
        products = batch['product_text']
        labels = batch['label'].to(device)
        weights = batch['weight'].to(device)
        
        # Forward pass
        query_embs = model.encode_queries(queries)
        product_embs = model.encode_products(products)
        
        # Compute loss
        loss, contrastive, inbatch = combined_loss(
            query_embs, product_embs, labels, weights
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Accumulate metrics
        loss_val = loss.item()
        contrastive_val = contrastive.item()
        inbatch_val = inbatch.item()
        
        total_loss += loss_val
        total_contrastive += contrastive_val
        total_inbatch += inbatch_val
        n_batches += 1
        steps_completed += 1
        
        # Update moving averages
        loss_window.append(loss_val)
        contrastive_window.append(contrastive_val)
        inbatch_window.append(inbatch_val)
        if len(loss_window) > window_size:
            loss_window.pop(0)
            contrastive_window.pop(0)
            inbatch_window.pop(0)
        
        # Save mid-epoch checkpoint every N batches (e.g., every 5000 batches)
        if checkpoint_dir is not None and (batch_idx + 1) % 5000 == 0:
            mid_checkpoint = {
                'epoch': epoch,
                'batch': batch_idx + 1,
                'stage': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'steps_completed': steps_completed,
                'is_mid_epoch': True
            }
            mid_checkpoint_path = checkpoint_dir / f"epoch_{epoch}_batch_{batch_idx+1}.pt"
            torch.save(mid_checkpoint, mid_checkpoint_path)
            print(f"    Saved mid-epoch checkpoint: batch {batch_idx+1}")
        
        # Log every 100 batches with moving average
        if (batch_idx + 1) % 100 == 0:
            # Calculate moving averages
            avg_loss = sum(loss_window) / len(loss_window)
            avg_contrastive = sum(contrastive_window) / len(contrastive_window)
            avg_inbatch = sum(inbatch_window) / len(inbatch_window)
            
            print(f"    Batch {batch_idx+1}/{len(train_loader)}, "
                  f"Loss: {loss_val:.4f} (avg: {avg_loss:.4f}), "
                  f"Contrastive: {contrastive_val:.4f} (avg: {avg_contrastive:.4f}), "
                  f"InBatch: {inbatch_val:.4f} (avg: {avg_inbatch:.4f})")
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    avg_contrastive = total_contrastive / n_batches if n_batches > 0 else 0.0
    avg_inbatch = total_inbatch / n_batches if n_batches > 0 else 0.0
    
    return {
        'train_loss': avg_loss,
        'train_contrastive': avg_contrastive,
        'train_inbatch': avg_inbatch
    }, steps_completed


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    print("=" * 80)
    print("TWO-TOWER MODEL - STAGE 1 TRAINING (KAGGLE)")
    print("=" * 80)
    
    # Configuration
    DATA_DIR = "/kaggle/input/search-and-retrieval-ttsn-training-data"
    CHECKPOINT_DIR = "/kaggle/working/checkpoints"
    BATCH_SIZE = 64
    NUM_WORKERS = 2  # Kaggle typically has 2 workers
    STAGE = 1
    EPOCHS = [1, 2]
    LEARNING_RATE = 2e-5
    BASE_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Embedding dimension: {EMBEDDING_DIM}")
    print(f"  Epochs: {EPOCHS}")
    
    # Create checkpoint directory
    Path(CHECKPOINT_DIR).mkdir(exist_ok=True, parents=True)
    
    # Load data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    
    train_df = pd.read_parquet(f"{DATA_DIR}/training_pairs.parquet")
    val_df = pd.read_parquet(f"{DATA_DIR}/val_pairs.parquet")
    
    print(f"  Train: {len(train_df):,} examples")
    print(f"  Val: {len(val_df):,} examples")
    
    # Create datasets
    print(f"\n{'='*80}")
    print("CREATING DATASETS")
    print(f"{'='*80}")
    
    train_dataset = TwoTowerDataset(train_df, stage=STAGE)
    val_dataset = TwoTowerDataset(val_df, stage=STAGE)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    print(f"\n{'='*80}")
    print("INITIALIZING MODEL")
    print(f"{'='*80}")
    
    model = TwoTowerModel(
        base_model_name=BASE_MODEL,
        embedding_dim=EMBEDDING_DIM,
        shared_encoder=True,
        normalize=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Learning rate scheduler with warmup (create before checkpoint loading)
    total_steps = len(train_loader) * len(EPOCHS)
    warmup_steps = max(1, int(total_steps * 0.1))
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"  Total training steps: {total_steps:,}")
    print(f"  Warmup steps: {warmup_steps:,}")
    
    # Check for existing checkpoints to resume
    print(f"\n{'='*80}")
    print("CHECKING FOR EXISTING CHECKPOINTS")
    print(f"{'='*80}")
    
    checkpoint_dir = Path(CHECKPOINT_DIR)
    existing_checkpoints = []
    mid_epoch_checkpoints = []
    
    # Check for epoch-end checkpoints
    for epoch in EPOCHS:
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
        if checkpoint_path.exists():
            existing_checkpoints.append(epoch)
    
    # Check for mid-epoch checkpoints
    for epoch in EPOCHS:
        # Look for mid-epoch checkpoints (format: epoch_X_batch_Y.pt)
        for checkpoint_file in checkpoint_dir.glob(f"epoch_{epoch}_batch_*.pt"):
            try:
                # Extract batch number from filename
                batch_num = int(checkpoint_file.stem.split('_batch_')[1])
                mid_epoch_checkpoints.append((epoch, batch_num, checkpoint_file))
            except:
                pass
    
    steps_completed = 0  # Track total steps for scheduler
    start_batch = 0  # Batch to start from (0 = start of epoch)
    resume_epoch = None
    start_epoch = EPOCHS[0]
    best_val_loss = float('inf')
    
    # Check for mid-epoch checkpoint first (more recent)
    if mid_epoch_checkpoints:
        # Get the most recent mid-epoch checkpoint
        latest_mid = max(mid_epoch_checkpoints, key=lambda x: (x[0], x[1]))
        resume_epoch, resume_batch, mid_checkpoint_path = latest_mid
        
        print(f"  Found mid-epoch checkpoint: {mid_checkpoint_path}")
        print(f"  Epoch {resume_epoch}, Batch {resume_batch}")
        
        # Load checkpoint
        checkpoint = torch.load(mid_checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded model state")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  ✓ Loaded optimizer state")
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  ✓ Loaded scheduler state")
        
        # Calculate steps completed
        if 'steps_completed' in checkpoint:
            steps_completed = checkpoint['steps_completed']
        else:
            # Calculate: (epoch - 1) * batches_per_epoch + batch
            steps_completed = (resume_epoch - 1) * len(train_loader) + resume_batch
        
        print(f"  Steps already completed: {steps_completed:,}")
        
        # Resume from this batch in this epoch
        start_epoch = resume_epoch
        start_batch = resume_batch
        EPOCHS_TO_TRAIN = [e for e in EPOCHS if e >= start_epoch]
        
        print(f"  Will resume from epoch {start_epoch}, batch {start_batch}")
        
        # Step scheduler to match completed steps (if not loaded from checkpoint)
        if 'scheduler_state_dict' not in checkpoint or checkpoint['scheduler_state_dict'] is None:
            for _ in range(steps_completed):
                scheduler.step()
            print(f"  Scheduler stepped to match {steps_completed:,} completed steps")
    
    elif existing_checkpoints:
        # Resume from latest checkpoint
        latest_epoch = max(existing_checkpoints)
        checkpoint_path = checkpoint_dir / f"epoch_{latest_epoch}.pt"
        
        print(f"  Found checkpoint: {checkpoint_path}")
        print(f"  Resuming from epoch {latest_epoch}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded model state from epoch {checkpoint['epoch']}")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  ✓ Loaded optimizer state")
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  ✓ Loaded scheduler state")
        
        # Calculate steps already completed
        # Steps per epoch * epochs completed
        steps_completed = len(train_loader) * latest_epoch
        print(f"  Steps already completed: {steps_completed:,}")
        
        # Get best validation loss
        if 'metrics' in checkpoint and 'val_loss' in checkpoint['metrics']:
            best_val_loss = checkpoint['metrics']['val_loss']
            print(f"  Previous best validation loss: {best_val_loss:.4f}")
        
        # Start from next epoch
        start_epoch = latest_epoch + 1
        print(f"  Will continue training from epoch {start_epoch}")
        
        # Filter epochs to train
        EPOCHS_TO_TRAIN = [e for e in EPOCHS if e >= start_epoch]
        
        if not EPOCHS_TO_TRAIN:
            print(f"\n  All epochs already completed!")
            print(f"  Training already finished. Checkpoints available:")
            for epoch in EPOCHS:
                if epoch in existing_checkpoints:
                    print(f"    - epoch_{epoch}.pt")
            return
    else:
        print(f"  No existing checkpoints found. Starting fresh training.")
        EPOCHS_TO_TRAIN = EPOCHS
        steps_completed = 0
        start_epoch = EPOCHS[0]
    
    # Training loop
    print(f"\n{'='*80}")
    print("STAGE 1: INITIAL TRAINING")
    print("=" * 80)
    print("Data: E+S positives + Random negatives")
    print("Goal: Learn basic query-product matching")
    if start_epoch > EPOCHS[0]:
        print(f"RESUMING: Starting from epoch {start_epoch}")
    print(f"{'='*80}")
    
    for epoch in EPOCHS_TO_TRAIN:
        print(f"\n{'='*80}")
        print(f"STAGE 1 - EPOCH {epoch}")
        print(f"{'='*80}")
        
        # Train
        # Only use start_batch for the first epoch we're resuming
        current_start_batch = start_batch if epoch == start_epoch else 0
        train_metrics, steps_completed = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, 
            steps_completed, checkpoint_dir=checkpoint_dir, 
            start_batch=current_start_batch
        )
        # Reset start_batch after first epoch
        start_batch = 0
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"\n  Training Loss: {train_metrics['train_loss']:.4f}")
        print(f"    Contrastive: {train_metrics['train_contrastive']:.4f}")
        print(f"    InBatch: {train_metrics['train_inbatch']:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'stage': STAGE,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': {
                **train_metrics,
                'val_loss': val_loss,
                'lr': current_lr
            }
        }
        
        checkpoint_path = Path(CHECKPOINT_DIR) / f"epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"\n  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = Path(CHECKPOINT_DIR) / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model: {best_path}")
    
    print(f"\n{'='*80}")
    print("STAGE 1 TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"\nCheckpoints saved to: {CHECKPOINT_DIR}")
    print(f"  - epoch_1.pt")
    print(f"  - epoch_2.pt")
    print(f"  - best_model.pt")
    
    # Zip checkpoints for easy download
    print(f"\n{'='*80}")
    print("CREATING CHECKPOINT ARCHIVE")
    print(f"{'='*80}")
    
    zip_path = Path("/kaggle/working/checkpoints_stage1.zip")
    checkpoint_dir = Path(CHECKPOINT_DIR)
    
    if checkpoint_dir.exists():
        # Create zip file
        shutil.make_archive(
            str(zip_path).replace('.zip', ''),  # Remove .zip extension (make_archive adds it)
            'zip',
            checkpoint_dir
        )
        print(f"\n✓ Checkpoint archive created: {zip_path}")
        print(f"  Size: {zip_path.stat().st_size / (1024*1024):.2f} MB")
        
        # List files in archive
        print(f"\n  Files included:")
        for checkpoint_file in sorted(checkpoint_dir.glob("*.pt")):
            size_mb = checkpoint_file.stat().st_size / (1024*1024)
            print(f"    - {checkpoint_file.name} ({size_mb:.2f} MB)")
    else:
        print(f"  Warning: Checkpoint directory not found: {CHECKPOINT_DIR}")
    
    print(f"\n{'='*80}")
    print("DOWNLOAD INSTRUCTIONS")
    print(f"{'='*80}")
    print(f"\nTo download checkpoints:")
    print(f"  1. Go to the 'Output' tab in your Kaggle notebook")
    print(f"  2. Find 'checkpoints_stage1.zip' in the file list")
    print(f"  3. Click the download button (⬇️) next to the zip file")
    print(f"\nAlternatively:")
    print(f"  1. Click 'Save Version' → 'Save & Run All'")
    print(f"  2. After it completes, go to 'Output' tab")
    print(f"  3. Download the zip file from there")
    print(f"\nNext steps after downloading:")
    print(f"  1. Extract checkpoints_stage1.zip")
    print(f"  2. Use epoch_2.pt for hard negative mining")
    print(f"  3. Train Stage 2 with new hard negatives")


if __name__ == "__main__":
    main()
