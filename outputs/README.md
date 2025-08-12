# Outputs Directory

Thư mục này chứa tất cả outputs, artifacts và cache từ training và evaluation.

## 📁 Cấu trúc thư mục

### 🤖 `/checkpoints` - Model checkpoints

- Model weights (.pth files)
- Training state (optimizer, scheduler)
- Best model checkpoints
- Learning curves (PNG files)

### 📊 `/logs` - Training logs

- Training progress logs
- Error logs
- Performance metrics logs
- Timestamped log files

### 💾 `/cache` - Data cache

- HuggingFace datasets cache
- Preprocessed data cache
- Streaming data buffers

### 🎯 `/artifacts` - W&B artifacts

- W&B run artifacts
- Exported model artifacts
- Experiment tracking data

### 📈 `/results` - Evaluation results

- Model comparison results
- Performance metrics
- Visualization plots
- Analysis reports

### 🔧 `/models` - Final models

- Production-ready models
- Exported models (ONNX, TorchScript)
- Model metadata

## 🚫 Gitignore

Toàn bộ thư mục `outputs/` được ignore trong Git để:

- Tránh commit large files (models, data)
- Giữ repository clean và lightweight
- Ngăn việc commit sensitive data

## 🧹 Cleanup

Để clean up outputs:

```bash
# Clean tất cả outputs
rm -rf outputs/*

# Clean chỉ cache
rm -rf outputs/cache/*

# Clean logs cũ (giữ 7 ngày gần nhất)
find outputs/logs -name "*.log" -mtime +7 -delete

# Clean checkpoints trung gian (giữ best models)
find outputs/checkpoints -name "*epoch*.pth" -delete
```

## 📋 Conventions

### Naming Convention

**Checkpoints:**

- `{model_name}_epoch_{num}.pth` - Regular checkpoints
- `{model_name}_best.pth` - Best model
- `{model_name}_final.pth` - Final model

**Logs:**

- `{model_name}_{timestamp}.log` - Training logs
- `evaluation_{timestamp}.log` - Evaluation logs

**Results:**

- `{model_name}_results.json` - Metrics
- `{model_name}_plots.png` - Visualizations

### Directory Usage

- **Development**: Sử dụng tất cả subdirectories
- **Production**: Chỉ sử dụng `/models` và `/results`
- **CI/CD**: Chỉ `/artifacts` cho deployment

## 🔧 Configuration

Các paths được config trong:

- `configs/project_config.yaml`
- `scripts/train.sh`
- `src/imgae_dx/cli/train.py`

Default paths:

- Checkpoints: `outputs/checkpoints`
- Logs: `outputs/logs`
- Cache: `outputs/cache`
