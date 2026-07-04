# NSGA-II + Successive Halving for Resource-Aware Chest X-Ray Fine-Tuning

This repository contains the implementation used for the paper:

**Multi-Objective Fine-Tuning Optimization for Pneumonia Detection Using Evolutionary Search and Successive Halving**

The project evaluates resource-aware fine-tuning configurations for binary chest X-ray pneumonia classification. The main optimization method combines **NSGA-II** with **Successive Halving** and searches for configurations that balance validation AUC, evaluation time, and trainable parameter count.

## Repository contents

├── config.json                  # Main experiment configuration
├── run_paper_reproduction.sh    # Full reproduction entry point
├── requirements.txt             # Runtime dependencies
├── pyproject.toml               # Package metadata and test configuration
├── experiments/                 # Experiment orchestration and analysis entry points
├── finetune_ga/                 # Core implementation
└── tests/                       # Tests

The experiments were developed for Python 3.10--3.12 and TensorFlow 2.19.0. The reported runs were executed on Kaggle with two NVIDIA Tesla T4 GPUs.
