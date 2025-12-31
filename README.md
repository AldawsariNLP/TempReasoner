# TempReasoner: Neural Temporal Graph Networks for Event Timeline Construction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

TempReasoner is a neural temporal graph network architecture for automated event timeline construction. It integrates multi-scale temporal attention, adaptive graph construction, hierarchical temporal encoding (GRU + Transformer), and reinforcement learning-based temporal reasoning within a unified framework.

**Key Features:**
- 94.3% timeline ordering accuracy on benchmark datasets
- Real-time inference with 127ms average latency
- Multi-scale temporal attention for capturing both short-term and long-term dependencies
- Adaptive graph construction for discovering implicit temporal relationships
- Temporal consistency loss for maintaining chronological coherence

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Dataset Preparation](#dataset-preparation)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Reproducing Experiments](#reproducing-experiments)
7. [Pre-trained Models](#pre-trained-models)
8. [Configuration](#configuration)
9. [Hardware Requirements](#hardware-requirements)
10. [Troubleshooting](#troubleshooting)
11. [Citation](#citation)

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)

### Step-by-Step Installation

```bash
# Clone the repository
git clone https://github.com/AldawsariNLP/TempReasoner.git
cd TempReasoner

# Create a virtual environment (recommended)
python -m venv tempreasoner_env
source tempreasoner_env/bin/activate  # On Windows: tempreasoner_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Dependencies

```txt
# requirements.txt
torch>=2.0.0
torch-geometric>=2.3.0
transformers>=4.30.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
tqdm>=4.65.0
wandb>=0.15.0  # Optional: for experiment tracking
tensorboard>=2.13.0
pyyaml>=6.0
```

## Quick Start

```python
from tempreasoner import TempReasoner, TemporalDataLoader

# Load pre-trained model
model = TempReasoner.from_pretrained('tempreasoner-base')

# Example: Construct timeline from events
events = [
    {"text": "Contract signed", "timestamp": "2024-01-15"},
    {"text": "Payment received", "timestamp": "2024-02-01"},
    {"text": "Project initiated", "timestamp": "2024-01-20"}
]

timeline = model.construct_timeline(events)
print(timeline)
# Output: [('Contract signed', '2024-01-15'), ('Project initiated', '2024-01-20'), ('Payment received', '2024-02-01')]
```

## Dataset Preparation

### Supported Datasets

| Dataset | Description | Download |
|---------|-------------|----------|
| ICEWS14 | Political events 2014 | [Link](https://huggingface.co/datasets/linxy/ICEWS14) |
| ICEWS05-15 | Political events 2005-2015 | [Link](https://github.com/stmrdus/TKGC-Benchmark-Datasets) |
| ICEWS18 | Political events 2018 | [Link](https://www.dgl.ai/dgl_docs/generated/dgl.data.ICEWS18Dataset.html) |
| GDELT | Global news events | [Link](https://www.kaggle.com/datasets/gdelt/gdelt) |
| TimeBank | Temporal annotations | [Link](https://www.kaggle.com/datasets/thanh097/timebench) |

### Download and Preprocess

```bash
# Download all datasets
python scripts/download_datasets.py --all

# Or download specific dataset
python scripts/download_datasets.py --dataset icews14

# Preprocess datasets
python scripts/preprocess.py --dataset icews14 --output_dir data/processed/
```

### Data Format

Events should be formatted as (subject, relation, object, timestamp) tuples:

```json
{
    "train": [
        {"subject": "USA", "relation": "visits", "object": "China", "timestamp": "2014-01-15"},
        ...
    ],
    "valid": [...],
    "test": [...]
}
```

## Training

### Basic Training

```bash
# Train on ICEWS14
python train.py --config configs/icews14.yaml

# Train on GDELT
python train.py --config configs/gdelt.yaml

# Train with custom parameters
python train.py \
    --dataset icews14 \
    --batch_size 64 \
    --learning_rate 2e-4 \
    --epochs 50 \
    --embedding_dim 200 \
    --num_heads 4 \
    --output_dir outputs/icews14_experiment
```

### Training Configuration

```yaml
# configs/icews14.yaml
model:
  embedding_dim: 200
  num_attention_heads: 4
  gru_hidden_dim: 256
  transformer_layers: 4
  dropout: 0.1

training:
  batch_size: 64
  learning_rate: 2e-4
  weight_decay: 1e-5
  epochs: 50
  warmup_steps: 1000
  gradient_clip: 1.0
  
loss:
  lambda_order: 1.0
  lambda_causality: 0.8
  lambda_transitivity: 0.6
  lambda_rl: 0.3

data:
  dataset: icews14
  train_split: 0.8
  valid_split: 0.1
  test_split: 0.1
```

### Multi-GPU Training

```bash
# Distributed training on 4 GPUs
torchrun --nproc_per_node=4 train.py --config configs/icews14.yaml --distributed
```

## Evaluation

### Run Evaluation

```bash
# Evaluate on test set
python evaluate.py \
    --model_path outputs/icews14_experiment/best_model.pt \
    --dataset icews14 \
    --split test

# Evaluate with all metrics
python evaluate.py \
    --model_path outputs/icews14_experiment/best_model.pt \
    --dataset icews14 \
    --metrics mrr hits@1 hits@3 hits@10 accuracy consistency
```

### Expected Output

```
=== Evaluation Results ===
Dataset: ICEWS14
Split: test

Temporal Link Prediction:
  MRR: 0.512
  Hits@1: 0.423
  Hits@3: 0.534
  Hits@10: 0.687

Timeline Ordering:
  Accuracy: 94.3%
  Temporal Consistency: 91.8%
  Causal Accuracy: 91.4%

Efficiency:
  Inference Latency: 127ms/sequence
  Memory Usage: 2.1GB
```

## Reproducing Experiments

### Full Reproduction Pipeline

```bash
# Step 1: Download and preprocess all datasets
./scripts/setup_experiments.sh

# Step 2: Run all experiments (5 seeds each)
python scripts/run_all_experiments.py --seeds 42 123 456 789 1024

# Step 3: Generate tables and figures
python scripts/generate_results.py --output_dir results/

# Step 4: Run ablation studies
python scripts/run_ablations.py --config configs/ablation.yaml
```

### Individual Experiment Commands

```bash
# Experiment 1: Main benchmark results (Table 2-4)
python train.py --config configs/icews14.yaml --seed 42
python train.py --config configs/icews18.yaml --seed 42
python train.py --config configs/gdelt.yaml --seed 42
python train.py --config configs/timebank.yaml --seed 42

# Experiment 2: Ablation studies (Table 6)
python train.py --config configs/ablation/no_multiscale.yaml
python train.py --config configs/ablation/no_adaptive_graph.yaml
python train.py --config configs/ablation/no_hierarchical.yaml
python train.py --config configs/ablation/no_rl.yaml
python train.py --config configs/ablation/no_consistency_loss.yaml

# Experiment 3: Cross-domain transfer (Table 8)
python scripts/cross_domain_transfer.py --source icews14 --target gdelt

# Experiment 4: Computational efficiency (Figure 6)
python scripts/benchmark_efficiency.py --max_sequence_length 500
```

### Expected Runtime

| Dataset | Training Time | GPU Memory | 
|---------|--------------|------------|
| ICEWS14 | ~8 hours | 2.1 GB |
| ICEWS18 | ~10 hours | 2.3 GB |
| GDELT | ~18 hours | 3.8 GB |
| TimeBank | ~6 hours | 1.8 GB |

*Times measured on single NVIDIA A100 40GB GPU*

## Pre-trained Models

### Available Models

| Model | Dataset | MRR | Download |
|-------|---------|-----|----------|
| tempreasoner-icews14 | ICEWS14 | 0.512 | [Link](https://github.com/AldawsariNLP/TempReasoner/releases) |
| tempreasoner-icews18 | ICEWS18 | 0.498 | [Link](https://github.com/AldawsariNLP/TempReasoner/releases) |
| tempreasoner-gdelt | GDELT | 0.223 | [Link](https://github.com/AldawsariNLP/TempReasoner/releases) |
| tempreasoner-timebank | TimeBank | 0.412 | [Link](https://github.com/AldawsariNLP/TempReasoner/releases) |

### Loading Pre-trained Models

```python
from tempreasoner import TempReasoner

# Load from Hugging Face Hub
model = TempReasoner.from_pretrained('AldawsariNLP/tempreasoner-icews14')

# Load from local checkpoint
model = TempReasoner.load_checkpoint('outputs/icews14/best_model.pt')
```

## Configuration

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 200 | Entity/relation embedding dimension |
| `num_attention_heads` | 4 | Multi-head attention heads |
| `gru_hidden_dim` | 256 | GRU hidden state dimension |
| `transformer_layers` | 4 | Number of transformer layers |
| `dropout` | 0.1 | Dropout probability |
| `temporal_scales` | [1, 7, 30, 365] | Temporal granularities (days) |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 2e-4 | Initial learning rate |
| `batch_size` | 64 | Training batch size |
| `epochs` | 50 | Maximum training epochs |
| `weight_decay` | 1e-5 | L2 regularization |
| `warmup_steps` | 1000 | Learning rate warmup |

## Hardware Requirements

### Minimum Requirements
- CPU: 8 cores
- RAM: 32 GB
- GPU: NVIDIA GPU with 8GB VRAM
- Storage: 50 GB free space

### Recommended Requirements
- CPU: 16+ cores (AMD EPYC or Intel Xeon)
- RAM: 64 GB
- GPU: NVIDIA A100 40GB
- Storage: 100 GB SSD

### Memory Scaling

| Batch Size | Sequence Length | GPU Memory |
|------------|-----------------|------------|
| 32 | 500 | 2.1 GB |
| 64 | 500 | 4.2 GB |
| 128 | 500 | 8.1 GB |

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train.py --batch_size 32

# Enable gradient checkpointing
python train.py --gradient_checkpointing

# Use mixed precision training
python train.py --fp16
```

**2. Slow Training**
```bash
# Enable multi-worker data loading
python train.py --num_workers 8

# Use compiled model (PyTorch 2.0+)
python train.py --compile
```

**3. Dataset Loading Errors**
```bash
# Verify dataset integrity
python scripts/verify_datasets.py --dataset icews14

# Re-download corrupted files
python scripts/download_datasets.py --dataset icews14 --force
```

**4. Reproducibility Issues**
```bash
# Set all random seeds
python train.py --seed 42 --deterministic

# Disable CUDA non-deterministic operations
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/AldawsariNLP/TempReasoner/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AldawsariNLP/TempReasoner/discussions)
- **Email**: [contact@example.com]

## Citation

If you use TempReasoner in your research, please cite:

```bibtex
@article{tempreasoner2025,
  title={TempReasoner: Neural Temporal Graph Networks for Event Timeline Construction},
  author={Aldawsari, et al.},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025},
  doi={10.1109/TNNLS.2025.XXXXXXX}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Datasets: ICEWS, GDELT, TimeBank
- Baseline implementations: TComplEx, TeMP, HyTE, RE-NET
- Computing resources: [Your institution]

---

**TempReasoner** - Advancing Temporal Reasoning in AI
