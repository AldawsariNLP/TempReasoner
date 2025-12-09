"""
TempReasoner: Neural Temporal Graph Networks for Event Timeline Construction
Advanced Model Development with Training and Testing Framework

This implementation includes:
- Multi-scale temporal attention mechanisms
- Adaptive graph construction layer
- Hierarchical temporal encoder (GRU + Transformer hybrid)
- Reinforcement learning-based timeline optimization
- Novel temporal consistency loss functions
- Comprehensive evaluation metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import math
from collections import namedtuple, defaultdict
from dataclasses import dataclass
import random
from datetime import datetime, timedelta
import logging 


# Configure logging ## file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for TempReasoner model"""
    # Embedding dimensions
    semantic_dim: int = 128
    temporal_dim: int = 64
    hidden_dim: int = 256
    
    # Architecture parameters
    num_gru_layers: int = 2
    num_transformer_layers: int = 2
    num_attention_heads: int = 4
    num_temporal_scales: int = 4
    
    # Training parameters
    learning_rate: float = 0.001
    rl_learning_rate: float = 0.0005
    batch_size: int = 32
    num_epochs: int = 100
    
    # Loss weights
    lambda_order: float = 1.0
    lambda_causality: float = 0.8
    lambda_transitivity: float = 0.6
    lambda_consistency: float = 1.0
    lambda_rl: float = 0.5
    
    # Temporal parameters
    max_sequence_length: int = 500
    margin: float = 0.5
    epsilon: float = 0.1
    
    # Graph construction
    adaptive_graph_threshold: float = 0.3
    
    # Dropout
    dropout_rate: float = 0.3


Transition = namedtuple('Transition', 
                       ('state', 'action', 'reward', 'next_state', 'done'))


class MultiScaleTemporalAttention(nn.Module):
    """Multi-scale temporal attention mechanism"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_scales = config.num_temporal_scales
        self.hidden_dim = config.hidden_dim
        
        # Create multi-scale attention heads
        self.scale_projections = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim // config.num_temporal_scales)
            for _ in range(config.num_temporal_scales)
        ])
        
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                config.hidden_dim // config.num_temporal_scales,
                num_heads=1,
                dropout=config.dropout_rate
            )
            for _ in range(config.num_temporal_scales)
        ])
        
        self.scale_weights = nn.Parameter(
            torch.ones(config.num_temporal_scales) / config.num_temporal_scales
        )
        
    def forward(self, x: torch.Tensor, temporal_distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            temporal_distances: (batch_size, seq_len, seq_len) - temporal distance matrix
        Returns:
            (batch_size, seq_len, hidden_dim) - multi-scale attended features
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Process at each temporal scale
        scale_outputs = []
        for scale_idx in range(self.num_scales):
            # Project to scale-specific dimension
            scale_feat = self.scale_projections[scale_idx](x)  # (B, L, D/S)
            
            # Compute temporal bias for this scale
            scale_factor = 2 ** scale_idx  # exponential scaling
            temporal_bias = temporal_distances / scale_factor
            temporal_bias = F.softmax(-temporal_bias, dim=-1)
            
            # Apply multi-head attention
            attn_out, _ = self.scale_attentions[scale_idx](
                scale_feat, scale_feat, scale_feat
            )
            
            # Apply temporal bias
            attn_out = attn_out * temporal_bias.unsqueeze(-1)
            scale_outputs.append(attn_out)
        
        # Concatenate and weight scale outputs
        scale_outputs = torch.cat(scale_outputs, dim=-1)  # (B, L, hidden_dim)
        
        return scale_outputs


class AdaptiveGraphConstructionLayer(nn.Module):
    """Dynamically constructs temporal graph with adaptive topology"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Similarity function parameters
        self.similarity_weight = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.similarity_bias = nn.Parameter(torch.zeros(1))
        
        # Temporal distance scaling
        self.temporal_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, features: torch.Tensor, 
                temporal_markers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch_size, seq_len, hidden_dim)
            temporal_markers: (batch_size, seq_len) - timestamps
        Returns:
            adjacency_matrix: (batch_size, seq_len, seq_len)
            edge_weights: (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, hidden_dim = features.shape
        
        # Compute semantic similarity
        # (B, L, 1, D) @ (B, 1, L, D) -> (B, L, L)
        semantic_sim = torch.bmm(
            self.similarity_weight(features),  # (B, L, D)
            features.transpose(1, 2)  # (B, D, L)
        ) / math.sqrt(hidden_dim)  # (B, L, L)
        
        # Compute temporal distance
        temporal_expanded_i = temporal_markers.unsqueeze(2)  # (B, L, 1)
        temporal_expanded_j = temporal_markers.unsqueeze(1)  # (B, 1, L)
        temporal_dist = torch.abs(temporal_expanded_i - temporal_expanded_j)  # (B, L, L)
        
        # Combine semantic similarity and temporal proximity
        combined = torch.tanh(semantic_sim) + \
                   -self.temporal_scale * temporal_dist / (temporal_dist.max() + 1e-8)
        
        # Apply sigmoid to get adaptive adjacency
        adjacency_matrix = torch.sigmoid(combined)
        
        # Threshold to create sparse graph
        edge_weights = adjacency_matrix.clone()
        adjacency_matrix = (adjacency_matrix > self.config.adaptive_graph_threshold).float()
        
        return adjacency_matrix, edge_weights


class HierarchicalTemporalEncoder(nn.Module):
    """Hierarchical encoder combining GRU local and Transformer global processing"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # GRU-based local temporal processor
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_gru_layers,
            dropout=config.dropout_rate if config.num_gru_layers > 1 else 0,
            batch_first=True
        )
        
        # Temporal gating mechanism
        self.temporal_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.Sigmoid()
        )
        
        # Transformer-based global temporal processor
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_transformer_layers
        )
        
        # Multi-scale attention fusion
        self.fusion_weights = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.Softmax(dim=-1)
        )
        
        self.cross_scale_interaction = nn.BilinearModule(
            config.hidden_dim, config.hidden_dim, config.hidden_dim
        ) if hasattr(nn, 'BilinearModule') else nn.Bilinear(
            config.hidden_dim, config.hidden_dim, config.hidden_dim
        )
        
    def forward(self, x: torch.Tensor, 
                temporal_distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            temporal_distances: (batch_size, seq_len, seq_len)
        Returns:
            (batch_size, seq_len, hidden_dim) - hierarchical encoding
        """
        # Local temporal processing (GRU)
        gru_out, _ = self.gru(x)  # (B, L, H)
        
        # Apply temporal gating
        gating_input = torch.cat([x, gru_out], dim=-1)
        temporal_gate = self.temporal_gate(gating_input)
        gru_gated = gru_out * temporal_gate
        
        # Global temporal processing (Transformer)
        transformer_out = self.transformer(x)  # (B, L, H)
        
        # Cross-scale interaction
        cross_interaction = self.cross_scale_interaction(
            gru_gated, transformer_out
        )  # (B, L, H)
        
        # Fusion
        fusion_input = torch.cat([gru_gated, transformer_out, cross_interaction], dim=-1)
        fusion_weights = self.fusion_weights(fusion_input)  # (B, L, H)
        
        fused = (gru_gated * fusion_weights[:, :, :self.config.hidden_dim] +
                 transformer_out * fusion_weights[:, :, self.config.hidden_dim:2*self.config.hidden_dim] +
                 cross_interaction * fusion_weights[:, :, 2*self.config.hidden_dim:])
        
        return fused


class TemporalConsistencyLoss(nn.Module):
    """Novel temporal consistency loss function"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.bce_loss = nn.BCELoss()
        
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-component consistency loss
        
        Args:
            predictions: dict with 'timestamps', 'causal_probs'
            targets: dict with 'timestamps', 'causal_labels', 'precedence'
        Returns:
            dict with 'order', 'causality', 'transitivity', 'total'
        """
        pred_times = predictions['timestamps']  # (B, L)
        pred_causal = predictions['causal_probs']  # (B, L, L)
        
        target_times = targets['timestamps']  # (B, L)
        target_causal = targets['causal_labels']  # (B, L, L)
        target_precedence = targets['precedence']  # (B, L, L)
        
        # Ordering loss
        batch_size, seq_len = pred_times.shape
        time_diff = pred_times.unsqueeze(2) - pred_times.unsqueeze(1)  # (B, L, L)
        target_order = (target_times.unsqueeze(2) < target_times.unsqueeze(1)).float()
        
        ordering_loss = torch.nn.functional.relu(
            self.config.margin - time_diff * target_order
        ).mean()
        
        # Causality loss
        causality_loss = self.bce_loss(pred_causal, target_causal.float())
        
        # Transitivity loss
        # For all i, j, k: if i < j < k, then pred_time[i] < pred_time[k]
        transitivity_violations = 0
        epsilon = self.config.epsilon
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                for k in range(j + 1, seq_len):
                    if target_precedence[0, i, k] > 0:  # i precedes k
                        violation = torch.nn.functional.relu(
                            epsilon - (pred_times[:, k] - pred_times[:, i])
                        )
                        transitivity_violations += violation.mean()
        
        transitivity_loss = transitivity_violations / max(1, seq_len * (seq_len - 1) * (seq_len - 2) / 6)
        
        total_loss = (self.config.lambda_order * ordering_loss +
                      self.config.lambda_causality * causality_loss +
                      self.config.lambda_transitivity * transitivity_loss)
        
        return {
            'order': ordering_loss,
            'causality': causality_loss,
            'transitivity': transitivity_loss,
            'total': total_loss
        }


class ReplayBuffer:
    """Experience replay buffer for RL component"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, transition: Transition):
        """Save a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample random batch of transitions"""
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self) -> int:
        return len(self.memory)


class TemporalReasoningAgent(nn.Module):
    """Deep RL agent for temporal reasoning"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 10)  # action space size
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch_size, hidden_dim * 3)
        Returns:
            action_logits: (batch_size, action_space)
            value: (batch_size, 1)
        """
        encoded = self.state_encoder(state)
        action_logits = self.policy_head(encoded)
        value = self.value_head(encoded)
        return action_logits, value


class TempReasoner(nn.Module):
    """Main TempReasoner architecture"""
    
    def __init__(self, config: ModelConfig, vocab_size: int = 5000):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.semantic_embedding = nn.Embedding(vocab_size, config.semantic_dim)
        self.temporal_embedding = nn.Embedding(100, config.temporal_dim)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.semantic_dim + config.temporal_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(
            config.max_sequence_length, config.hidden_dim
        )
        
        # Core components
        self.multi_scale_attention = MultiScaleTemporalAttention(config)
        self.adaptive_graph = AdaptiveGraphConstructionLayer(config)
        self.hierarchical_encoder = HierarchicalTemporalEncoder(config)
        self.consistency_loss_fn = TemporalConsistencyLoss(config)
        
        # Timeline generation
        self.timestamp_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
        self.causal_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # RL agent
        self.rl_agent = TemporalReasoningAgent(config)
        self.replay_buffer = ReplayBuffer()
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        self.rl_optimizer = optim.Adam(self.rl_agent.parameters(), 
                                       lr=config.rl_learning_rate)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, event_tokens: torch.Tensor,
                temporal_markers: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            event_tokens: (batch_size, seq_len)
            temporal_markers: (batch_size, seq_len)
        Returns:
            dict with predictions
        """
        batch_size, seq_len = event_tokens.shape
        device = event_tokens.device
        
        # Embedding
        semantic_emb = self.semantic_embedding(event_tokens)
        temporal_indices = (temporal_markers % 100).long()
        temporal_emb = self.temporal_embedding(temporal_indices)
        
        # Feature fusion
        combined = torch.cat([semantic_emb, temporal_emb], dim=-1)
        features = self.feature_fusion(combined)  # (B, L, H)
        
        # Add positional encoding
        pe = self.positional_encoding[:seq_len, :].unsqueeze(0).to(device)
        features = features + pe
        
        # Compute temporal distances
        temporal_distances = torch.abs(
            temporal_markers.unsqueeze(2) - temporal_markers.unsqueeze(1)
        ).float()  # (B, L, L)
        
        # Multi-scale attention
        attended_features = self.multi_scale_attention(features, temporal_distances)
        
        # Adaptive graph construction
        adjacency_matrix, edge_weights = self.adaptive_graph(
            attended_features, temporal_markers
        )
        
        # Graph convolution with adaptive weights
        graph_features = torch.bmm(adjacency_matrix, attended_features)
        
        # Hierarchical encoding
        encoded = self.hierarchical_encoder(graph_features, temporal_distances)
        
        # Predict timestamps
        predicted_times = self.timestamp_predictor(encoded).squeeze(-1)  # (B, L)
        
        # Predict causal relationships
        causal_logits = []
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    pair_features = torch.cat(
                        [encoded[:, i, :], encoded[:, j, :]], dim=-1
                    )
                    causal_logits.append(self.causal_predictor(pair_features))
        
        if causal_logits:
            causal_probs = torch.stack(causal_logits, dim=1)  # (B, L*L, 1)
            causal_probs = causal_probs.reshape(batch_size, seq_len, seq_len)
        else:
            causal_probs = torch.zeros(batch_size, seq_len, seq_len, device=device)
        
        return {
            'timestamps': predicted_times,
            'causal_probs': causal_probs,
            'features': encoded,
            'adjacency': adjacency_matrix,
            'edge_weights': edge_weights
        }
    
    def compute_rl_state(self, predictions: Dict[str, torch.Tensor],
                        temporal_markers: torch.Tensor) -> torch.Tensor:
        """Compute state for RL agent"""
        features = predictions['features']  # (B, L, H)
        batch_size, seq_len, hidden_dim = features.shape
        
        # Timeline consistency metric
        pred_times = predictions['timestamps']
        consistency_score = self._compute_consistency(pred_times, temporal_markers)
        
        # Uncertainty metric
        temporal_uncertainty = pred_times.std(dim=1, keepdim=True)
        
        # Current timeline configuration
        timeline_state = features.mean(dim=1)  # (B, H)
        
        # Concatenate: [timeline_state, consistency, uncertainty]
        state = torch.cat([
            timeline_state,
            consistency_score.unsqueeze(-1),
            temporal_uncertainty
        ], dim=-1)  # (B, H+2)
        
        return state
    
    def _compute_consistency(self, pred_times: torch.Tensor,
                            target_times: torch.Tensor) -> torch.Tensor:
        """Compute timeline consistency score"""
        batch_size = pred_times.shape[0]
        consistency_scores = []
        
        for b in range(batch_size):
            pred_ordered = torch.argsort(pred_times[b])
            target_ordered = torch.argsort(target_times[b])
            
            # Compute Spearman correlation-like metric
            correlation = (pred_ordered == target_ordered).float().mean()
            consistency_scores.append(correlation)
        
        return torch.tensor(consistency_scores, device=pred_times.device)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self(batch['event_tokens'], batch['temporal_markers'])
        
        # Compute losses
        consistency_losses = self.consistency_loss_fn(
            predictions,
            {
                'timestamps': batch['target_times'],
                'causal_labels': batch['causal_labels'],
                'precedence': batch['precedence']
            }
        )
        
        # Total supervised loss
        supervised_loss = consistency_losses['total']
        
        # RL loss
        rl_state = self.compute_rl_state(predictions, batch['temporal_markers'])
        action_logits, value_pred = self.rl_agent(rl_state)
        
        # Dummy RL reward for now (should come from environment)
        rl_reward = consistency_losses['total'].detach()
        rl_loss = F.smooth_l1_loss(value_pred.squeeze(), rl_reward)
        
        # Total loss
        total_loss = (supervised_loss + 
                     self.config.lambda_consistency * consistency_losses['total'] +
                     self.config.lambda_rl * rl_loss)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'supervised_loss': supervised_loss.item(),
            'order_loss': consistency_losses['order'].item(),
            'causality_loss': consistency_losses['causality'].item(),
            'transitivity_loss': consistency_losses['transitivity'].item(),
            'rl_loss': rl_loss.item()
        }
    
    @torch.no_grad()
    def test_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single test step with metrics"""
        self.eval()
        
        # Forward pass
        predictions = self(batch['event_tokens'], batch['temporal_markers'])
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, batch)
        
        return metrics
    
    def _compute_metrics(self, predictions: Dict[str, torch.Tensor],
                         batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute evaluation metrics"""
        pred_times = predictions['timestamps']
        target_times = batch['target_times']
        pred_causal = predictions['causal_probs']
        target_causal = batch['causal_labels']
        
        metrics = {}
        
        # Timeline ordering accuracy
        pred_order = torch.argsort(pred_times, dim=1)
        target_order = torch.argsort(target_times, dim=1)
        ordering_accuracy = (pred_order == target_order).float().mean()
        metrics['ordering_accuracy'] = ordering_accuracy.item()
        
        # Temporal consistency score
        batch_size, seq_len = pred_times.shape
        consistency_scores = []
        for b in range(batch_size):
            pred_pairs = []
            target_pairs = []
            for i in range(seq_len):
                for j in range(i+1, seq_len):
                    pred_pairs.append(pred_times[b, i] < pred_times[b, j])
                    target_pairs.append(target_times[b, i] < target_times[b, j])
            
            if pred_pairs:
                pred_pairs = torch.stack(pred_pairs)
                target_pairs = torch.stack(target_pairs)
                consistency = (pred_pairs == target_pairs).float().mean()
                consistency_scores.append(consistency.item())
        
        metrics['temporal_consistency'] = np.mean(consistency_scores) if consistency_scores else 0.0
        
        # Causal relationship accuracy
        pred_causal_binary = (pred_causal > 0.5).float()
        causal_accuracy = (pred_causal_binary == target_causal).float().mean()
        metrics['causal_accuracy'] = causal_accuracy.item()
        
        # MAE for timestamps
        mae = torch.abs(pred_times - target_times).mean()
        metrics['timestamp_mae'] = mae.item()
        
        return metrics


def create_synthetic_dataset(num_samples: int = 100,
                            seq_length: int = 20,
                            vocab_size: int = 5000) -> Tuple[List[Dict], List[Dict]]:
    """Create synthetic temporal event dataset"""
    train_data = []
    test_data = []
    
    for _ in range(num_samples):
        # Random event tokens
        event_tokens = torch.randint(1, vocab_size, (seq_length,))
        
        # Generate realistic temporal markers (timestamps)
        base_time = 0
        temporal_markers = []
        for i in range(seq_length):
            temporal_markers.append(base_time)
            base_time += random.uniform(1, 10)  # Random interval between events
        temporal_markers = torch.tensor(temporal_markers, dtype=torch.float32)
        
        # Create target timestamps (normalized)
        target_times = temporal_markers.clone()
        
        # Causal labels (random for now)
        causal_labels = torch.randint(0, 2, (seq_length, seq_length)).float()
        causal_labels = (causal_labels + causal_labels.t()) / 2  # Make symmetric-ish
        
        # Precedence
        precedence = torch.zeros(seq_length, seq_length)
        for i in range(seq_length):
            for j in range(i+1, seq_length):
                if target_times[i] < target_times[j]:
                    precedence[i, j] = 1
                else:
                    precedence[j, i] = 1
        
        sample = {
            'event_tokens': event_tokens,
            'temporal_markers': temporal_markers,
            'target_times': target_times,
            'causal_labels': causal_labels,
            'precedence': precedence
        }
        
        if _ < int(0.8 * num_samples):
            train_data.append(sample)
        else:
            test_data.append(sample)
    
    return train_data, test_data


def create_data_loader(dataset: List[Dict], batch_size: int = 32, shuffle: bool = True):
    """Create data loader"""
    class TempDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset_obj = TempDataset(dataset)
    return torch.utils.data.DataLoader(
        dataset_obj, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )


def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch samples"""
    # Pad sequences to same length
    max_len = max(len(sample['event_tokens']) for sample in batch)
    
    event_tokens_padded = []
    temporal_markers_padded = []
    target_times_padded = []
    
    for sample in batch:
        seq_len = len(sample['event_tokens'])
        pad_len = max_len - seq_len
        
        event_tokens_padded.append(
            F.pad(sample['event_tokens'].unsqueeze(0), (0, pad_len), value=0)
        )
        temporal_markers_padded.append(
            F.pad(sample['temporal_markers'].unsqueeze(0), (0, pad_len), value=0)
        )
        target_times_padded.append(
            F.pad(sample['target_times'].unsqueeze(0), (0, pad_len), value=0)
        )
    
    return {
        'event_tokens': torch.cat(event_tokens_padded, dim=0).long(),
        'temporal_markers': torch.cat(temporal_markers_padded, dim=0),
        'target_times': torch.cat(target_times_padded, dim=0),
        'causal_labels': torch.stack([s['causal_labels'] for s in batch]),
        'precedence': torch.stack([s['precedence'] for s in batch])
    }


def train_epoch(model: TempReasoner, train_loader, device: str = 'cpu') -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    epoch_losses = defaultdict(float)
    num_batches = 0
    
    for batch_data in train_loader:
        # Move to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in collate_batch([batch_data]).items()}
        
        # Train step
        losses = model.train_step(batch)
        
        for key, value in losses.items():
            epoch_losses[key] += value
        num_batches += 1
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
    return avg_losses


def evaluate(model: TempReasoner, test_loader, device: str = 'cpu') -> Dict[str, float]:
    """Evaluate model"""
    model.eval()
    epoch_metrics = defaultdict(float)
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in test_loader:
            # Move to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in collate_batch([batch_data]).items()}
            
            # Test step
            metrics = model.test_step(batch)
            
            for key, value in metrics.items():
                epoch_metrics[key] += value
            num_batches += 1
    
    # Average metrics
    avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
    return avg_metrics


if __name__ == "__main__":
    # Configuration
    config = ModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {config}")
    
    # Create model
    model = TempReasoner(config)
    model = model.to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create dataset
    logger.info("Creating synthetic dataset...")
    train_data, test_data = create_synthetic_dataset(num_samples=100)
    
    # Create data loaders
    train_loader = create_data_loader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = create_data_loader(test_data, batch_size=config.batch_size, shuffle=False)
    
    # Training loop
    logger.info("Starting training...")
    best_accuracy = 0.0
    
    for epoch in range(config.num_epochs):
        train_losses = defaultdict(float)
        
        for batch_data in train_loader:
            batch = collate_batch([batch_data])
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            losses = model.train_step(batch)
            for k, v in losses.items():
                train_losses[k] += v
        
        # Average losses
        num_batches = len(train_loader)
        avg_train_losses = {k: v / num_batches for k, v in train_losses.items()}
        
        # Evaluation
        if (epoch + 1) % 10 == 0:
            test_metrics = evaluate(model, test_loader, device)
            
            logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
            logger.info(f"Train Loss: {avg_train_losses['total_loss']:.4f}")
            logger.info(f"Test Ordering Accuracy: {test_metrics['ordering_accuracy']:.4f}")
            logger.info(f"Test Consistency: {test_metrics['temporal_consistency']:.4f}")
            logger.info(f"Test Causal Accuracy: {test_metrics['causal_accuracy']:.4f}")
            
            if test_metrics['ordering_accuracy'] > best_accuracy:
                best_accuracy = test_metrics['ordering_accuracy']
                torch.save(model.state_dict(), '/mnt/user-data/outputs/best_model.pt')
    
    logger.info(f"\nTraining completed! Best accuracy: {best_accuracy:.4f}")
    logger.info("Model saved to /mnt/user-data/outputs/best_model.pt")
