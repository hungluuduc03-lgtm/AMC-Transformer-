# AMC-Transformer

A novel transformer-based framework for Automatic Modulation Classification (AMC) that directly processes raw I/Q time series data.

## Key Features
-  **High Accuracy**: 98.8% accuracy on RadioML2018.01A (SNR ≥ 10 dB)
-  **End-to-End**: Direct processing of raw I/Q signals without handcrafted features
-  **Self-Attention**: Multi-head attention mechanism for capturing long-range dependencies
-  **Robust**: Consistent performance across different SNR conditions

## Abstract
This repository implements the AMC-Transformer model described in our paper "AMC-Transformer: Automatic Modulation Classification based on Enhanced Attention Model". The model tokenizes raw I/Q sequences into fixed-length patches, augments them with learnable positional embeddings, and applies multi-layer, multi-head self-attention to capture global temporal-spatial correlations.
