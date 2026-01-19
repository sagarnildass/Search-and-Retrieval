"""
Two-Tower Model Architecture for Query-Product Matching

This module implements:
- QueryTower: Encodes queries into embeddings
- ProductTower: Encodes products into embeddings
- TwoTowerModel: Combined model with similarity computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import Optional, Tuple


class QueryTower(nn.Module):
    """Query encoder tower"""
    
    def __init__(
        self,
        base_model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        normalize: bool = True
    ):
        """
        Args:
            base_model_name: Name of the SentenceTransformer model
            embedding_dim: Output embedding dimension
            normalize: Whether to L2 normalize embeddings
        """
        super(QueryTower, self).__init__()
        
        self.base_encoder = SentenceTransformer(base_model_name)
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        
        # Freeze base encoder initially (can be unfrozen later)
        # We'll fine-tune it during training
        
    def forward(self, query_texts):
        """
        Encode queries into embeddings with gradient support
        
        Args:
            query_texts: List of query strings or single string
            
        Returns:
            Query embeddings [batch_size, embedding_dim]
        """
        # Handle single string input
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
        """
        Args:
            base_model_name: Name of the SentenceTransformer model
            embedding_dim: Output embedding dimension
            normalize: Whether to L2 normalize embeddings
            shared_encoder: If provided, share encoder with query tower
        """
        super(ProductTower, self).__init__()
        
        if shared_encoder is not None:
            # Share encoder with query tower
            self.base_encoder = shared_encoder
        else:
            # Separate encoder
            self.base_encoder = SentenceTransformer(base_model_name)
        
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        
    def forward(self, product_texts):
        """
        Encode products into embeddings with gradient support
        
        Args:
            product_texts: List of product text strings or single string
            
        Returns:
            Product embeddings [batch_size, embedding_dim]
        """
        # Handle single string input
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
        """
        Args:
            base_model_name: Name of the SentenceTransformer model
            embedding_dim: Output embedding dimension
            shared_encoder: Whether to share encoder between towers
            normalize: Whether to normalize embeddings
        """
        super(TwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        
        # Create query tower
        self.query_tower = QueryTower(
            base_model_name=base_model_name,
            embedding_dim=embedding_dim,
            normalize=normalize
        )
        
        # Create product tower
        if shared_encoder:
            # Share encoder
            self.product_tower = ProductTower(
                base_model_name=base_model_name,
                embedding_dim=embedding_dim,
                normalize=normalize,
                shared_encoder=self.query_tower.base_encoder
            )
        else:
            # Separate encoder
            self.product_tower = ProductTower(
                base_model_name=base_model_name,
                embedding_dim=embedding_dim,
                normalize=normalize,
                shared_encoder=None
            )
    
    def forward(
        self,
        query_texts,
        product_texts,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            query_texts: List of query strings
            product_texts: List of product text strings
            return_embeddings: Whether to return embeddings
            
        Returns:
            similarity_scores: [batch_size] similarity scores
            query_embs: [batch_size, embedding_dim] if return_embeddings=True
            product_embs: [batch_size, embedding_dim] if return_embeddings=True
        """
        # Encode queries and products
        query_embs = self.query_tower(query_texts)
        product_embs = self.product_tower(product_texts)
        
        # Compute cosine similarity (dot product when normalized)
        similarity = torch.sum(query_embs * product_embs, dim=1)
        
        if return_embeddings:
            return similarity, query_embs, product_embs
        else:
            return similarity
    
    def encode_queries(self, query_texts):
        """Encode queries only"""
        return self.query_tower(query_texts)
    
    def encode_products(self, product_texts):
        """Encode products only"""
        return self.product_tower(product_texts)
