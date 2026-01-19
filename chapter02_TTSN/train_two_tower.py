"""
Training Script for Two-Tower Model with Stage Management

This script implements:
- Stage-based training (Stage 1, 2, 3)
- Checkpointing and resuming
- Hard negative mining integration
- Validation and metrics tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import argparse
from typing import Dict, Optional, Tuple
import time
import warnings

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from two_tower_model import TwoTowerModel
from losses import combined_loss

warnings.filterwarnings('ignore')


class TwoTowerDataset(Dataset):
    """Dataset that filters by training stage"""
    
    def __init__(self, df: pd.DataFrame, stage: int = 1):
        """
        Args:
            df: DataFrame with training pairs (must have use_in_stage_X columns)
            stage: Training stage (1, 2, or 3)
        """
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


class StageManager:
    """Manages training stages and transitions"""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.current_stage = 1
        self.current_epoch = 0
    
    def get_stage_config(self, stage: int) -> Dict:
        """Get configuration for a specific stage"""
        configs = {
            1: {
                'epochs': [1, 2],
                'lr': 2e-5,
                'include_c': False,
                'include_hard_negatives': False,
                'description': 'Initial Training (E+S positives, random negatives)'
            },
            2: {
                'epochs': [3],
                'lr': 2e-5,  # Same as Stage 1
                'include_c': False,
                'include_hard_negatives': True,
                'description': 'Hard Negative Training (E+S positives, random + hard negatives)'
            },
            3: {
                'epochs': [4, 5],
                'lr': 2e-6,  # 10x reduction
                'include_c': True,
                'include_hard_negatives': True,
                'description': 'Fine-tuning (E+S+C positives, all negatives)'
            }
        }
        return configs[stage]
    
    def should_mine_hard_negatives(self, epoch: int) -> bool:
        """Check if we should mine hard negatives after this epoch"""
        # Mine after epoch 2 (end of Stage 1)
        return epoch == 2
    
    def get_checkpoint_path(self, epoch: int) -> Path:
        """Get checkpoint path for an epoch"""
        return self.checkpoint_dir / f"epoch_{epoch}.pt"
    
    def get_best_model_path(self) -> Path:
        """Get path for best model"""
        return self.checkpoint_dir / "best_model.pt"
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        stage: int,
        metrics: Dict,
        is_best: bool = False
    ):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics
        }
        
        # Save epoch checkpoint
        checkpoint_path = self.get_checkpoint_path(epoch)
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.get_best_model_path()
            torch.save(checkpoint, best_path)
            print(f"  Saved best model: {best_path}")
    
    def load_checkpoint(self, epoch: int) -> Optional[Dict]:
        """Load checkpoint"""
        checkpoint_path = self.get_checkpoint_path(epoch)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"  Loaded checkpoint: epoch {epoch}")
            return checkpoint
        return None


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> float:
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
            
            # Forward pass
            query_embs = model.encode_queries(queries)
            product_embs = model.encode_products(products)
            
            # Compute loss
            loss, _, _ = combined_loss(
                query_embs, product_embs, labels, weights
            )
            
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
    epoch: int
) -> Dict:
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
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"  Epoch {epoch}")):
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
        
        # Update moving averages
        loss_window.append(loss_val)
        contrastive_window.append(contrastive_val)
        inbatch_window.append(inbatch_val)
        if len(loss_window) > window_size:
            loss_window.pop(0)
            contrastive_window.pop(0)
            inbatch_window.pop(0)
        
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
    }


def train_stage(
    model: nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    stage: int,
    stage_config: Dict,
    stage_manager: StageManager,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4
) -> nn.Module:
    """Train model for a specific stage"""
    
    print("\n" + "=" * 80)
    print(f"STAGE {stage}: {stage_config['description']}")
    print("=" * 80)
    
    # Create datasets
    train_dataset = TwoTowerDataset(train_df, stage=stage)
    val_dataset = TwoTowerDataset(val_df, stage=stage)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Setup optimizer with stage-specific learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stage_config['lr'],
        weight_decay=0.01
    )
    
    # Learning rate scheduler with warmup (create before checkpoint loading)
    # Calculate total steps for this stage
    epochs_in_stage = len(stage_config['epochs'])
    total_steps = len(train_loader) * epochs_in_stage
    warmup_steps = max(1, int(total_steps * 0.1))
    
    # Use get_linear_schedule_with_warmup style scheduler
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"  Total training steps for stage: {total_steps:,}")
    print(f"  Warmup steps: {warmup_steps:,}")
    
    # Load checkpoint if resuming
    start_epoch = stage_config['epochs'][0]
    if start_epoch > 1:
        checkpoint = stage_manager.load_checkpoint(start_epoch - 1)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"  ✓ Loaded scheduler state")
            
            print(f"  Resumed from epoch {start_epoch - 1}")
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in stage_config['epochs']:
        print(f"\n{'='*80}")
        print(f"STAGE {stage} - EPOCH {epoch}")
        print(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        
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
        metrics = {
            **train_metrics,
            'val_loss': val_loss,
            'lr': current_lr
        }
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        stage_manager.save_checkpoint(
            model, optimizer, scheduler, epoch, stage, metrics, is_best=is_best
        )
        
        # Check if we need to mine hard negatives
        if stage_manager.should_mine_hard_negatives(epoch):
            print(f"\n{'='*80}")
            print("HARD NEGATIVE MINING REQUIRED")
            print(f"{'='*80}")
            print("  After Stage 1 completes, run hard_negative_mining.py")
            print("  to generate hard negatives for Stage 2")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Two-Tower Model')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing training data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory for checkpoints')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], default=None,
                       help='Specific stage to train (default: train all stages)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--base_model', type=str, default='all-MiniLM-L6-v2',
                       help='Base SentenceTransformer model')
    parser.add_argument('--embedding_dim', type=int, default=384,
                       help='Embedding dimension')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 80)
    print("TWO-TOWER MODEL TRAINING")
    print("=" * 80)
    print(f"\nDevice: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Base model: {args.base_model}")
    print(f"Embedding dimension: {args.embedding_dim}")
    
    # Load data
    data_dir = Path(args.data_dir)
    print(f"\nLoading data from: {data_dir}")
    
    train_df = pd.read_parquet(data_dir / "training_pairs.parquet")
    val_df = pd.read_parquet(data_dir / "val_pairs.parquet")
    
    print(f"  Train: {len(train_df):,} examples")
    print(f"  Val: {len(val_df):,} examples")
    
    # Initialize model
    print(f"\nInitializing model...")
    model = TwoTowerModel(
        base_model_name=args.base_model,
        embedding_dim=args.embedding_dim,
        shared_encoder=True,
        normalize=True
    ).to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize stage manager
    stage_manager = StageManager(args.checkpoint_dir)
    
    # Determine which stages to train
    if args.stage is not None:
        stages_to_train = [args.stage]
    else:
        stages_to_train = [1, 2, 3]
    
    # Train each stage
    for stage in stages_to_train:
        stage_config = stage_manager.get_stage_config(stage)
        stage_manager.current_stage = stage
        
        model = train_stage(
            model=model,
            train_df=train_df,
            val_df=val_df,
            stage=stage,
            stage_config=stage_config,
            stage_manager=stage_manager,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nBest model saved to: {stage_manager.get_best_model_path()}")


if __name__ == "__main__":
    main()
