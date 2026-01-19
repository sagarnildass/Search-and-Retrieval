"""
Loss Functions for Two-Tower Model Training

This module implements:
- Weighted Contrastive Loss
- In-Batch Negative Loss (InfoNCE)
- Combined Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def weighted_contrastive_loss(
    query_embs: torch.Tensor,
    product_embs: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    margin: float = 0.2
) -> torch.Tensor:
    """
    Weighted contrastive loss
    
    For positive pairs: minimize distance (weighted by relevance)
    For negative pairs: maximize distance (with margin)
    
    Args:
        query_embs: [batch_size, embedding_dim] Query embeddings
        product_embs: [batch_size, embedding_dim] Product embeddings
        labels: [batch_size] 1.0 for positive, 0.0 for negative
        weights: [batch_size] Relevance weights (1.0, 0.9, 0.6, or 0.0)
        margin: Margin for negative pairs
        
    Returns:
        Loss scalar
    """
    # Compute squared L2 distance (when normalized, this is 2 - 2*cosine_sim)
    # Since embeddings are normalized, ||q - p||² = 2 - 2*cos(q, p)
    distances = torch.sum((query_embs - product_embs) ** 2, dim=1)
    
    # Positive pairs: minimize distance (weighted)
    positive_mask = (labels == 1.0).float()
    positive_loss = (weights * distances * positive_mask).sum()
    n_positives = positive_mask.sum()
    
    # Negative pairs: maximize distance (with margin)
    # Loss = max(0, margin - distance)²
    negative_mask = (labels == 0.0).float()
    negative_distances = distances * negative_mask
    negative_loss = (torch.clamp(margin - negative_distances, min=0.0) ** 2).sum()
    n_negatives = negative_mask.sum()
    
    # Normalize by number of examples
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
    """
    In-batch negative loss (InfoNCE)
    
    Uses other products in the batch as negatives for each query.
    Maximizes similarity to positive product, minimizes similarity to negatives.
    
    Args:
        query_embs: [batch_size, embedding_dim] Query embeddings
        product_embs: [batch_size, embedding_dim] Product embeddings
        labels: [batch_size] 1.0 for positive, 0.0 for negative
        temperature: Temperature parameter for softmax
        
    Returns:
        Loss scalar
    """
    # Compute similarity matrix: [batch_size, batch_size]
    # similarity_matrix[i, j] = similarity(query_i, product_j)
    similarity_matrix = torch.matmul(query_embs, product_embs.T)  # [B, B]
    
    # Scale by temperature
    similarity_matrix = similarity_matrix / temperature
    
    # Positive pairs are on diagonal (query i matches product i)
    positive_scores = torch.diag(similarity_matrix)  # [B]
    
    # Compute log-softmax over all products (in-batch negatives)
    # For each query, we want to maximize similarity to its positive
    # and minimize similarity to all other products (negatives)
    log_probs = positive_scores - torch.logsumexp(similarity_matrix, dim=1)
    
    # Only compute loss for positive pairs
    positive_mask = (labels == 1.0).float()
    loss = -(log_probs * positive_mask).sum()
    
    # Normalize by number of positive pairs
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
) -> torch.Tensor:
    """
    Combined loss: weighted contrastive + in-batch negatives
    
    Args:
        query_embs: [batch_size, embedding_dim] Query embeddings
        product_embs: [batch_size, embedding_dim] Product embeddings
        labels: [batch_size] 1.0 for positive, 0.0 for negative
        weights: [batch_size] Relevance weights
        contrastive_weight: Weight for contrastive loss
        inbatch_weight: Weight for in-batch negative loss
        margin: Margin for contrastive loss
        temperature: Temperature for in-batch negative loss
        
    Returns:
        Combined loss scalar
    """
    # Compute both losses
    contrastive = weighted_contrastive_loss(
        query_embs, product_embs, labels, weights, margin
    )
    inbatch = in_batch_negative_loss(
        query_embs, product_embs, labels, temperature
    )
    
    # Combine
    total_loss = contrastive_weight * contrastive + inbatch_weight * inbatch
    
    return total_loss, contrastive, inbatch


def triplet_loss(
    anchor_embs: torch.Tensor,
    positive_embs: torch.Tensor,
    negative_embs: torch.Tensor,
    margin: float = 0.2
) -> torch.Tensor:
    """
    Triplet loss
    
    Pulls anchor and positive together, pushes anchor and negative apart.
    
    Args:
        anchor_embs: [batch_size, embedding_dim] Anchor (query) embeddings
        positive_embs: [batch_size, embedding_dim] Positive product embeddings
        negative_embs: [batch_size, embedding_dim] Negative product embeddings
        margin: Margin between positive and negative distances
        
    Returns:
        Loss scalar
    """
    # Compute distances
    pos_dist = torch.sum((anchor_embs - positive_embs) ** 2, dim=1)
    neg_dist = torch.sum((anchor_embs - negative_embs) ** 2, dim=1)
    
    # Triplet loss: max(0, pos_dist - neg_dist + margin)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    
    return loss.mean()
