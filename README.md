# Neural Archaeology: Decoding the Hidden Representations of Language Models

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)

Course project for **ECE4424/CS4824 Machine Learning** (Fall 2025, Prof. Ming Jin).

## Overview

Large language models are often treated as black boxes. This project examines the internal activations of **SmolLM-1.7B-Instruct** (24 layers, 2048-dimensional hidden states) to understand how it encodes high-level concepts such as **safety** (harmful vs. safe responses) and **emotion**.

By extracting hidden states layer by layer and analyzing them with both unsupervised and supervised methods, the project identifies where these concepts emerge in the network and how reliably they can be detected.

The work builds on representation engineering ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)) and the circuit breakers dataset ([Zou et al., 2024](https://www.circuit-breaker.ai/)).

## Methodology

| Section | Task | Techniques |
|---|---|---|
| Q1 | Pooling strategy comparison | Last, mean, max and first-token pooling; cosine similarity; within/between-class separation |
| Q2 | Dimensionality reduction | PCA implemented from scratch (covariance and eigendecomposition) |
| Q3 | Unsupervised emotion discovery | K-means implemented from scratch; ARI and silhouette score |
| Q4 | Natural concept dimensions | PCA on safety and emotion representations |
| Q5 | Supervised safety detection | Logistic regression probes across layers; ROC AUC |
| Q6 | Data efficiency | Learning curves comparing supervised probes against a PCA baseline |

## Key Results

- **Last-token pooling** gives the strongest class separation (mean separation 1.112), since it is the only position that attends to the full input in a causal model.
- Safety information is concentrated in the **middle-to-late layers**. The best linear probe reaches **94% accuracy and 0.973 AUC at layer 18**.
- **Supervised probes clearly outperform PCA** for safety detection (AUC 0.973 vs. 0.619). The two directions are nearly orthogonal (cosine similarity 0.07).
- Unsupervised methods (PCA, K-means) tend to capture **surface-level variation** rather than semantic categories (emotion ARI of 0.015).
- Around **50 labeled examples** are enough for a supervised probe to reliably beat the unsupervised baseline.

## Repository Structure

```
├── Neural_Archaeology_Martina.ipynb   # Main notebook: code, results and analysis
├── docs/                              # PDF export of the notebook
├── data/
│   ├── safety/                        # Circuit breakers dataset
│   └── emotions/                      # Emotion scenarios
├── results/                           # Generated figures and evaluations
├── assets/                            # Notebook images
├── environment.yml                    # Conda environment
└── requirements.txt                   # Pip dependencies
```

## Getting Started

```bash
git clone git@github.com:martinagg7/ML4424_NeuralArchaeology.git
cd ML4424_NeuralArchaeology

conda env create -f environment.yml
conda activate neural-archaeology

jupyter notebook Neural_Archaeology_Martina.ipynb
```

A CUDA-capable GPU is recommended. The model is downloaded automatically from Hugging Face on first run.
