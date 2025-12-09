"""
TempReasoner Advanced Training Framework
Includes comprehensive evaluation, logging, visualization, and deployment utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, roc_curve, auc
)
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict
from datetime import datetime
import time
from collections import deque
import logging

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track model performance metrics across training"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.epoch_metrics = []
        
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.metrics[key].append(value)
    
    def get_average(self, key: str) -> float:
        """Get average of metric"""
        if key in self.metrics and len(self.metrics[key]) > 0:
            return np.mean(self.metrics[key])
        return 0.0
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get all averaged metrics"""
        return {k: np.mean(v) for k, v in self.metrics.items() if len(v) > 0}
    
    def record_epoch(self, metrics: Dict[str, float]):
        """Record epoch metrics"""
        self.epoch_metrics.append({
            'timestamp': datetime.now().isoformat(),
            **metrics
        })
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        return pd.DataFrame(self.epoch_metrics)


class TemporalMetricsCalculator:
    """Calculate advanced temporal metrics"""
    
    @staticmethod
    def mean_reciprocal_rank(predictions: np.ndarray, 
                            targets: np.ndarray) -> float:
        """Calculate MRR metric"""
        mrr_sum = 0.0
        for pred, target in zip(predictions, targets):
            sorted_indices = np.argsort(-pred)
            rank = np.where(sorted_indices == target)[0][0] + 1
            mrr_sum += 1.0 / rank
        return mrr_sum / len(predictions)
    
    @staticmethod
    def hits_at_k(predictions: np.ndarray,
                  targets: np.ndarray,
                  k: int = 1) -> float:
        """Calculate Hits@K metric"""
        hits = 0
        for pred, target in zip(predictions, targets):
            top_k_indices = np.argsort(-pred)[:k]
            if target in top_k_indices:
                hits += 1
        return hits / len(predictions)
    
    @staticmethod
    def temporal_mae(pred_times: torch.Tensor,
                     target_times: torch.Tensor) -> float:
        """Mean Absolute Error for timestamps"""
        return torch.abs(pred_times - target_times).mean().item()
    
    @staticmethod
    def temporal_rmse(pred_times: torch.Tensor,
                      target_times: torch.Tensor) -> float:
        """Root Mean Squared Error for timestamps"""
        return torch.sqrt(torch.mean((pred_times - target_times) ** 2)).item()
    
    @staticmethod
    def chronological_correctness(pred_times: torch.Tensor,
                                 target_times: torch.Tensor) -> float:
        """Percentage of correctly ordered event pairs"""
        batch_size = pred_times.shape[0]
        correct = 0
        total = 0
        
        for b in range(batch_size):
            for i in range(len(pred_times[b])):
                for j in range(i + 1, len(pred_times[b])):
                    pred_order = pred_times[b, i] < pred_times[b, j]
                    target_order = target_times[b, i] < target_times[b, j]
                    
                    if pred_order == target_order:
                        correct += 1
                    total += 1
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def causal_f1_score(pred_causal: torch.Tensor,
                       target_causal: torch.Tensor) -> float:
        """F1 score for causal relationship prediction"""
        pred_binary = (pred_causal > 0.5).cpu().numpy()
        target_binary = target_causal.cpu().numpy()
        
        tp = np.sum(pred_binary * target_binary)
        fp = np.sum(pred_binary * (1 - target_binary))
        fn = np.sum((1 - pred_binary) * target_binary)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return f1


class ModelCheckpointer:
    """Save and load model checkpoints"""
    
    def __init__(self, save_dir: str = '/mnt/user-data/outputs/checkpoints'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_metrics = {}
    
    def save_checkpoint(self, model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
            self.best_metrics = metrics
    
    def load_checkpoint(self, model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       checkpoint_path: str) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return epoch


class ResultsVisualizer:
    """Visualize training results"""
    
    @staticmethod
    def plot_loss_curves(tracker: PerformanceTracker,
                        save_path: Optional[str] = None):
        """Plot training loss curves"""
        df = tracker.to_dataframe()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if 'total_loss' in df.columns:
            axes[0, 0].plot(df['total_loss'])
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        if 'order_loss' in df.columns:
            axes[0, 1].plot(df['order_loss'], label='Order Loss')
            axes[0, 1].set_title('Order Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        if 'causality_loss' in df.columns:
            axes[1, 0].plot(df['causality_loss'], label='Causality Loss')
            axes[1, 0].set_title('Causality Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        if 'rl_loss' in df.columns:
            axes[1, 1].plot(df['rl_loss'], label='RL Loss')
            axes[1, 1].set_title('Reinforcement Learning Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved loss curves to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_metric_curves(tracker: PerformanceTracker,
                          save_path: Optional[str] = None):
        """Plot metric curves"""
        df = tracker.to_dataframe()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = [
            ('ordering_accuracy', 'Ordering Accuracy'),
            ('temporal_consistency', 'Temporal Consistency'),
            ('causal_accuracy', 'Causal Accuracy'),
            ('timestamp_mae', 'Timestamp MAE')
        ]
        
        for idx, (metric_name, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            if metric_name in df.columns:
                ax.plot(df[metric_name], marker='o')
                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved metric curves to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(predictions: np.ndarray,
                             targets: np.ndarray,
                             save_path: Optional[str] = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved confusion matrix to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_roc_curves(predictions: np.ndarray,
                       targets: np.ndarray,
                       save_path: Optional[str] = None):
        """Plot ROC curves"""
        if len(np.unique(targets)) < 2:
            logger.warning("Cannot plot ROC curve with less than 2 classes")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        fpr, tpr, _ = roc_curve(targets, predictions)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved ROC curve to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_calibration_curve(confidence: np.ndarray,
                              accuracy: np.ndarray,
                              save_path: Optional[str] = None,
                              num_bins: int = 10):
        """Plot calibration curve"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        mean_confidence = []
        mean_accuracy = []
        
        for i in range(num_bins):
            mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])
            if mask.sum() > 0:
                mean_confidence.append(confidence[mask].mean())
                mean_accuracy.append(accuracy[mask].mean())
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(mean_confidence, mean_accuracy, 'o-', label='Model')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title('Calibration Curve')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved calibration curve to {save_path}")
        plt.close()


class ExperimentTracker:
    """Track and log experiments"""
    
    def __init__(self, exp_name: str,
                save_dir: str = '/mnt/user-data/outputs/experiments'):
        self.exp_name = exp_name
        self.save_dir = Path(save_dir) / exp_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_path = self.save_dir / 'config.json'
        self.metrics_path = self.save_dir / 'metrics.json'
        self.log_path = self.save_dir / 'experiment.log'
        
        # Setup logging
        handler = logging.FileHandler(self.log_path)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
    
    def save_config(self, config_dict: Dict):
        """Save configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Saved config to {self.config_path}")
    
    def save_metrics(self, metrics: Dict):
        """Save metrics"""
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {self.metrics_path}")
    
    def get_checkpoint_dir(self) -> Path:
        """Get checkpoint directory"""
        checkpoint_dir = self.save_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        return checkpoint_dir
    
    def get_visualization_dir(self) -> Path:
        """Get visualization directory"""
        viz_dir = self.save_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        return viz_dir


class AdvancedTrainer:
    """Advanced trainer with comprehensive evaluation"""
    
    def __init__(self, model: nn.Module,
                config,
                device: str = 'cpu',
                experiment_name: str = 'tempreasoner_exp'):
        self.model = model
        self.config = config
        self.device = device
        
        self.tracker = PerformanceTracker()
        self.metrics_calc = TemporalMetricsCalculator()
        self.visualizer = ResultsVisualizer()
        self.experiment = ExperimentTracker(experiment_name)
        self.checkpointer = ModelCheckpointer(
            str(self.experiment.get_checkpoint_dir())
        )
        
        # Save config
        self.experiment.save_config(asdict(config))
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        for batch_data in progress_bar:
            batch = self._prepare_batch(batch_data)
            
            losses = self.model.train_step(batch)
            
            for key, value in losses.items():
                epoch_losses[key] += value
            
            num_batches += 1
            progress_bar.update(1)
        
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        return avg_losses
    
    def evaluate_epoch(self, test_loader) -> Dict[str, float]:
        """Evaluate for one epoch"""
        self.model.eval()
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc='Evaluating')
            for batch_data in progress_bar:
                batch = self._prepare_batch(batch_data)
                
                metrics = self.model.test_step(batch)
                
                for key, value in metrics.items():
                    epoch_metrics[key] += value
                
                num_batches += 1
                progress_bar.update(1)
        
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        return avg_metrics
    
    def _prepare_batch(self, batch_data) -> Dict[str, torch.Tensor]:
        """Prepare batch for training"""
        if isinstance(batch_data, dict):
            batch = batch_data
        else:
            batch = collate_batch([batch_data])
        
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()}
        return batch
    
    def fit(self, train_loader,
           val_loader,
           num_epochs: int,
           early_stopping_patience: int = 10):
        """Fit model"""
        best_val_accuracy = 0
        patience_counter = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.evaluate_epoch(val_loader)
            
            # Update tracker
            self.tracker.update(**train_losses)
            self.tracker.update(**val_metrics)
            self.tracker.record_epoch({**train_losses, **val_metrics})
            
            # Log
            logger.info(f"Train Loss: {train_losses.get('total_loss', 0):.4f}")
            logger.info(f"Val Accuracy: {val_metrics.get('ordering_accuracy', 0):.4f}")
            logger.info(f"Val Consistency: {val_metrics.get('temporal_consistency', 0):.4f}")
            
            # Checkpoint
            is_best = val_metrics.get('ordering_accuracy', 0) > best_val_accuracy
            if is_best:
                best_val_accuracy = val_metrics.get('ordering_accuracy', 0)
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.checkpointer.save_checkpoint(
                self.model,
                self.model.optimizer,
                epoch,
                val_metrics,
                is_best=is_best
            )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Save results
        self.experiment.save_metrics(self.tracker.get_all_averages())
        
        # Visualize
        viz_dir = self.experiment.get_visualization_dir()
        self.visualizer.plot_loss_curves(
            self.tracker,
            str(viz_dir / 'loss_curves.png')
        )
        self.visualizer.plot_metric_curves(
            self.tracker,
            str(viz_dir / 'metric_curves.png')
        )
        
        logger.info(f"\nTraining completed! Best accuracy: {best_val_accuracy:.4f}")


# Import tqdm if available
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable, desc='', **kwargs):
            self.iterable = iterable
            self.desc = desc
        
        def __iter__(self):
            return iter(self.iterable)
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def update(self, n=1):
            pass


# Import defaultdict
from collections import defaultdict
