# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ImgAE-Dx is a medical image anomaly detection research project that compares different autoencoder architectures for unsupervised anomaly detection in medical images. The project focuses on comparing U-Net (baseline) with Reversed Autoencoder (RA) architectures for detecting abnormalities in chest X-rays.

## Core Architecture

This is a research-focused project implemented as Jupyter notebooks:

### Main Components

- **Anomaly_Detection_Research.ipynb**: Local development notebook
- **Anomaly_Detection_Research_Colab.ipynb**: Google Colab version with cloud integration and checkpointing

### Key Models

1. **U-Net**: Baseline model with skip connections for detailed reconstruction
2. **Reversed Autoencoder (RA)**: Experimental asymmetric architecture without skip connections, designed to create "pseudo-healthy" reconstructions

### Research Methodology

- **Unsupervised learning**: Models trained only on "normal" chest X-ray images
- **Anomaly detection**: Uses reconstruction error to identify abnormalities
- **Comparative analysis**: Direct performance comparison using AUC-ROC metrics

## Working with Notebooks

### Local Development

```bash
# Install Jupyter if not available
pip install jupyter

# Launch notebook
jupyter notebook Anomaly_Detection_Research.ipynb
```

### Google Colab Version

The Colab notebook includes:

- Automatic Google Drive integration for persistent storage
- Kaggle API integration for dataset download
- Checkpointing system for long training sessions
- GPU acceleration support

### Required Data Setup

- NIH Chest X-ray dataset from Kaggle
- Requires kaggle.json API key file
- Dataset filtering for "No Finding" (normal) vs abnormal cases

## Key Configuration Parameters

Located in notebook cell configuration:

```python
IMAGE_DIR = "/content/images"  # Dataset location
IMG_SIZE = 128                 # Input image dimensions
BATCH_SIZE = 32               # Training batch size
EPOCHS = 10                   # Training epochs (increase for better results)
LR = 1e-4                     # Learning rate
NUM_NORMAL_SAMPLES = 2000     # Normal samples for training
NUM_ABNORMAL_SAMPLES = 1000   # Abnormal samples for testing
```

## Development Workflow

### Training Process

1. **Data Preparation**: Filter and split NIH Chest X-ray dataset
2. **Model Training**: Train both U-Net and RA models on normal images only
3. **Evaluation**: Test on mixed normal/abnormal dataset using reconstruction error
4. **Analysis**: Compare AUC scores and visualize error maps

### Checkpointing System (Colab)

- Automatic model state saving after each epoch
- Includes optimizer state and training history
- Enables recovery from interrupted sessions
- Stores checkpoints in Google Drive for persistence

## Performance Evaluation

### Metrics Used

- **AUC-ROC**: Primary quantitative metric for anomaly classification
- **Reconstruction Error Maps**: Qualitative analysis of anomaly localization
- **Loss Curves**: Training convergence monitoring

### Expected Outcomes

- U-Net typically shows superior performance due to skip connections
- RA may show better anomaly localization in specific cases
- Both models should achieve AUC > 0.80 on test data

## Research Context

Based on paper: "Towards Universal Unsupervised Anomaly Detection in Medical Imaging"

- Source: <https://arxiv.org/abs/2401.10637v1>
- Implementation reference: <https://github.com/ci-ber/RA>

## Dependencies

### Core Libraries (automatically installed in Colab)

- PyTorch (deep learning framework)
- torchvision (image transforms)
- scikit-learn (evaluation metrics)
- matplotlib (visualization)
- pandas (data handling)
- PIL/Pillow (image processing)
- tqdm (progress bars)

### Data Requirements

- Kaggle account and API key
- ~45GB disk space for full NIH dataset (subset used for demo)
- GPU recommended for training (available in Colab)

## File Structure Context

- `REQUIREMENT.md`: Vietnamese assignment requirements for autoencoder project
- `Structure.md`: Detailed execution plan and theoretical rationale
- `TODO.md`: Implementation checklist with deep learning concepts
- `IDEA.md`: Research paper summary and methodology
- `PROJECT_JOURNEY.md`: Research methodology and experimental design
- `QUESTION.md`: QEC framework for notebook development

## Tips for Working with This Project

1. **Always use the Colab version** for actual training due to GPU requirements and dataset size
2. **Set up checkpointing** before starting long training runs
3. **Monitor memory usage** when working with full datasets
4. **Adjust sample sizes** (NUM_NORMAL_SAMPLES, NUM_ABNORMAL_SAMPLES) based on available resources
5. **Increase EPOCHS** for production runs (20-30 recommended vs demo 10)
6. **Compare both models** on same data splits for fair evaluation
