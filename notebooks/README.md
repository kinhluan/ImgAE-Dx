# 📚 ImgAE-Dx Notebooks

This directory contains Jupyter notebooks for the ImgAE-Dx medical image anomaly detection project.

## 🚀 Available Notebooks

### **T4_GPU_Training_Colab.ipynb** - Production Training Notebook
**🎯 Primary notebook for Google Colab T4 GPU training**

#### Features:
- ✅ **T4 GPU Optimization**: Mixed precision training with 16GB VRAM efficiency
- ✅ **HuggingFace Streaming**: Memory-efficient dataset loading without local storage
- ✅ **Professional Checkpointing**: Google Drive backup and session recovery
- ✅ **Advanced Monitoring**: W&B experiment tracking and performance analysis
- ✅ **Script Integration**: Uses optimized training scripts with T4 detection
- ✅ **Complete Workflow**: Setup → Training → Evaluation → Results

#### Performance Expectations:
- **Training Speed**: ~850 samples/sec (with mixed precision)
- **Memory Usage**: 12-14GB / 16GB T4 VRAM (75-85% utilization)
- **Training Time**: 
  - Quick test (1.5K samples, 10 epochs): 20-25 minutes
  - Standard (3K samples, 20 epochs): 45-60 minutes
  - Research quality (5K samples, 30 epochs): 75-90 minutes

#### Quick Start:
1. Open in Google Colab
2. Select T4 GPU runtime
3. Run all cells sequentially
4. Monitor progress in W&B dashboard

## 🔬 Research Context

### Methodology
Based on **"Towards Universal Unsupervised Anomaly Detection in Medical Imaging"**
- **Approach**: Unsupervised learning using reconstruction error
- **Models**: U-Net (baseline) vs Reversed Autoencoder comparison
- **Evaluation**: AUC-ROC, AUC-PR, F1-Score for anomaly detection

### Datasets Supported
| Dataset | Size | Samples | Speed | Description |
|---------|------|---------|-------|-------------|
| `keremberke/chest-xray-classification` | ~5GB | ~5,800 | Fast ✅ | Chest X-ray normal/pneumonia |
| `alkzar90/NIH-Chest-X-ray-dataset` | ~45GB | ~112K | Medium | NIH with 14 pathology labels |
| `Francesco/chest-xray-pneumonia-detection` | ~2GB | ~5,200 | Very Fast ✅ | Pneumonia detection |

## 🎯 Usage Instructions

### For Google Colab:
```python
# 1. Upload T4_GPU_Training_Colab.ipynb to Colab
# 2. Set Runtime → Change runtime type → T4 GPU
# 3. Run cells sequentially
# 4. Models automatically saved to Google Drive
```

### For Local Jupyter:
```bash
# Start Jupyter
jupyter notebook

# Open T4_GPU_Training_Colab.ipynb
# Note: Designed for Colab, may need modifications for local use
```

## 📊 Expected Outputs

### Training Artifacts:
- **Models**: Saved to `/content/drive/MyDrive/imgae_dx_checkpoints/`
- **Logs**: Training logs with performance metrics
- **Visualizations**: ROC curves, reconstruction error heatmaps
- **W&B Dashboard**: Real-time training monitoring

### Research Results:
- **Performance Comparison**: U-Net vs Reversed Autoencoder metrics
- **Anomaly Detection**: AUC scores for medical image anomalies
- **Architecture Analysis**: Parameter count, memory usage, training speed
- **Error Analysis**: Reconstruction quality and anomaly localization

## 🔧 Troubleshooting

### Common Issues:
1. **GPU OOM**: Reduce batch size from 48 → 32 → 16
2. **Slow training**: Verify T4 detection and mixed precision enabled
3. **Colab disconnection**: Training auto-saves every 2 epochs
4. **Dataset errors**: Try alternative datasets or check HF token

### Performance Optimization:
- **Batch Size**: 48 (T4 + AMP), 32 (T4 no AMP), 16 (conservative)
- **Memory Limit**: 14GB (safe), 15GB (aggressive), 12GB (conservative)
- **Workers**: 2 (optimal for T4), 1 (if memory issues)

## 🏆 Success Criteria

### Technical Validation:
- ✅ T4 GPU detected and optimized
- ✅ Mixed precision training enabled
- ✅ Models training with loss convergence
- ✅ Checkpoints saved to Google Drive
- ✅ W&B experiment tracking active

### Research Validation:
- ✅ Both model architectures implemented
- ✅ Anomaly detection metrics calculated
- ✅ Statistical comparison framework
- ✅ Reproducible results with seed management

## 📝 Notes for Researchers

### Best Practices:
1. **Start with quick test** (1.5K samples) to validate setup
2. **Monitor W&B dashboard** for real-time training metrics
3. **Save notebook to Drive** after completion
4. **Document hyperparameters** for reproducibility
5. **Compare multiple runs** for statistical significance

### Extension Opportunities:
- **Different architectures**: Add VAE, ResNet-based autoencoders
- **More datasets**: Expand to other medical imaging modalities
- **Advanced metrics**: Add SSIM, LPIPS for reconstruction quality
- **Ensemble methods**: Combine multiple model predictions
- **Transfer learning**: Pre-trained medical image encoders

## 🎉 Getting Started

**Ready to start your medical AI research on T4 GPU?**

1. **Upload** `T4_GPU_Training_Colab.ipynb` to Google Colab
2. **Select** T4 GPU runtime 
3. **Run** all cells sequentially
4. **Monitor** training progress and results
5. **Analyze** your trained models for medical anomaly detection

**Happy researching! 🧠🔬**