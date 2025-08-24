# 🔬 Hướng dẫn Evaluation cho CẢ 2 MODELS - Trả lời Research Questions

## 📋 **Mục đích**
Evaluation để trả lời TOÀN BỘ Research Questions trong PROJECT_JOURNEY.md:
- **RQ1**: U-Net có đạt AUC > 0.80 (baseline effectiveness)?
- **RQ2**: RA có performance tốt hơn U-Net không?
- **H1**: U-Net baseline effectiveness
- **H2**: RA specialized performance với better localization

## 🚀 **Thêm các cell sau vào cuối notebook (sau cell 20)**

---

## **Cell 21: Load Test Data (NORMAL + PNEUMONIA)**

```python
# ========================================
# EVALUATION PHASE: Load Balanced Test Data
# ========================================

from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, roc_curve
from scipy import stats
import numpy as np

print("🔄 Loading balanced test dataset (NORMAL + PNEUMONIA)...")

# Load full dataset để tạo test set cân bằng
full_dataset = load_dataset(
    CONFIG['hf_dataset'],
    CONFIG['hf_config'],
    split=CONFIG['hf_split'],
    **auth_kwargs
)

print(f"Full dataset size: {len(full_dataset)}")
print(f"Dataset features: {full_dataset.features}")

# Filter by labels
normal_data = full_dataset.filter(lambda x: x['label'] == 0)  # NORMAL
pneumonia_data = full_dataset.filter(lambda x: x['label'] == 1)  # PNEUMONIA

# Create balanced test set (500 each class)
test_size_per_class = 500
test_normal = normal_data.select(range(min(test_size_per_class, len(normal_data))))
test_pneumonia = pneumonia_data.select(range(min(test_size_per_class, len(pneumonia_data))))

print(f"✅ Test set: {len(test_normal)} NORMAL + {len(test_pneumonia)} PNEUMONIA")

# Process test data
def process_test_data(dataset_normal, dataset_pneumonia, transform):
    test_images = []
    test_labels = []
    
    print("Processing NORMAL images...")
    for i, item in enumerate(dataset_normal):
        image = item[CONFIG['image_column']]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        test_images.append(transform(image).unsqueeze(0))
        test_labels.append(0)  # NORMAL = 0
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(dataset_normal)} normal images")
    
    print("Processing PNEUMONIA images...")
    for i, item in enumerate(dataset_pneumonia):
        image = item[CONFIG['image_column']]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        test_images.append(transform(image).unsqueeze(0))
        test_labels.append(1)  # PNEUMONIA = 1
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(dataset_pneumonia)} pneumonia images")
    
    return torch.cat(test_images, dim=0), np.array(test_labels)

test_images, test_labels = process_test_data(test_normal, test_pneumonia, transform)

print(f"\\n✅ Test tensor shape: {test_images.shape}")
print(f"Label distribution: NORMAL={np.sum(test_labels==0)}, PNEUMONIA={np.sum(test_labels==1)}")
```

---

## **Cell 22: Load Both Models và Compute Reconstruction Errors**

```python
# ========================================
# Load Both Models and Compute Errors
# ========================================

def compute_reconstruction_error(model, images, batch_size=32):
    """Compute reconstruction error for anomaly detection"""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            reconstructed = model(batch)
            
            # Compute MSE per image
            mse_per_image = torch.mean((batch - reconstructed)**2, dim=[1,2,3])
            errors.extend(mse_per_image.cpu().numpy())
    
    return np.array(errors)

# Load both models for evaluation
eval_models = {}
model_errors = {}

# Load U-Net
if 'unet' in [name for name, _ in models_to_train] or 'unet' in trained_models:
    print("📂 Loading U-Net model...")
    unet_eval = UNet(
        in_channels=1,
        out_channels=1, 
        features=[64, 128, 256, 512]
    ).to(device)
    
    unet_path = f"{CONFIG['checkpoint_dir']}/unet_best.pth"
    try:
        unet_eval.load_state_dict(torch.load(unet_path, map_location=device))
        print(f"✅ U-Net loaded from: {unet_path}")
    except:
        print("⚠️ Using U-Net from memory (checkpoint not found)")
        unet_eval = trained_models.get('unet', None)
        if unet_eval is None:
            print("❌ U-Net not available for evaluation")
    
    if unet_eval is not None:
        eval_models['unet'] = unet_eval
        print("🔄 Computing U-Net reconstruction errors...")
        model_errors['unet'] = compute_reconstruction_error(unet_eval, test_images)
        print(f"✅ U-Net errors computed")

# Load Reversed AE  
if 'reversed_ae' in [name for name, _ in models_to_train] or 'reversed_ae' in trained_models:
    print("\\n📂 Loading Reversed Autoencoder model...")
    ra_eval = ReversedAutoencoder(
        in_channels=1,
        latent_dim=128,
        image_size=CONFIG['image_size']
    ).to(device)
    
    ra_path = f"{CONFIG['checkpoint_dir']}/reversed_ae_best.pth"
    try:
        ra_eval.load_state_dict(torch.load(ra_path, map_location=device))
        print(f"✅ RA loaded from: {ra_path}")
    except:
        print("⚠️ Using RA from memory (checkpoint not found)")
        ra_eval = trained_models.get('reversed_ae', None)
        if ra_eval is None:
            print("❌ RA not available for evaluation")
    
    if ra_eval is not None:
        eval_models['reversed_ae'] = ra_eval
        print("🔄 Computing RA reconstruction errors...")
        model_errors['reversed_ae'] = compute_reconstruction_error(ra_eval, test_images)
        print(f"✅ RA errors computed")

print(f"\\n📊 Models ready for evaluation: {list(eval_models.keys())}")
for model_name, errors in model_errors.items():
    print(f"   {model_name}: {len(errors)} samples, error range [{errors.min():.4f} - {errors.max():.4f}]")
```

---

## **Cell 23: Calculate AUC-ROC - ANSWER RQ1 & RQ2**

```python
# ========================================
# RESEARCH QUESTIONS VALIDATION
# ========================================

model_metrics = {}

print("🎯 RESEARCH QUESTIONS VALIDATION")
print("=" * 80)

# Calculate metrics for each model
for model_name, errors in model_errors.items():
    # Calculate AUC-ROC
    auc_roc = roc_auc_score(test_labels, errors)
    
    # Calculate AUC-PR
    precision, recall, _ = precision_recall_curve(test_labels, errors)
    auc_pr = np.trapz(recall, precision)
    
    # Find optimal threshold
    fpr, tpr, roc_thresholds = roc_curve(test_labels, errors)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = roc_thresholds[optimal_idx]
    
    # Calculate F1 at optimal threshold
    pred_labels = (errors > optimal_threshold).astype(int)
    f1 = f1_score(test_labels, pred_labels)
    
    # Error statistics by class
    normal_errors = errors[test_labels == 0]
    pneumonia_errors = errors[test_labels == 1]
    
    model_metrics[model_name] = {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'f1_score': f1,
        'optimal_threshold': optimal_threshold,
        'sensitivity': tpr[optimal_idx],
        'specificity': 1 - fpr[optimal_idx],
        'normal_error_mean': normal_errors.mean(),
        'pneumonia_error_mean': pneumonia_errors.mean(),
        'error_separation': pneumonia_errors.mean() - normal_errors.mean(),
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall
    }
    
    print(f"\\n📊 {model_name.upper()} RESULTS:")
    print(f"   AUC-ROC: {auc_roc:.4f}")
    print(f"   AUC-PR:  {auc_pr:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Sensitivity: {tpr[optimal_idx]:.4f}")
    print(f"   Specificity: {1-fpr[optimal_idx]:.4f}")
    print(f"   Error Separation: {pneumonia_errors.mean() - normal_errors.mean():.4f}")

print("\\n" + "="*80)

# ANSWER RQ1: U-Net Baseline Effectiveness
if 'unet' in model_metrics:
    unet_auc = model_metrics['unet']['auc_roc']
    print("🎯 **RESEARCH QUESTION RQ1 VALIDATION**")
    print(f"   Question: Can U-Net achieve AUC > 0.80 baseline?")
    print(f"   U-Net AUC-ROC: {unet_auc:.4f}")
    
    if unet_auc > 0.80:
        print("   ✅ **H1 CONFIRMED**: U-Net achieves effective baseline")
        print("   ✅ **RQ1 ANSWER: YES** - Skip connections enable effective anomaly detection")
    else:
        print("   ❌ **H1 REJECTED**: U-Net below 0.80 threshold")
        print("   ❌ **RQ1 ANSWER: NO** - Baseline not effective enough")

# ANSWER RQ2: RA vs U-Net Comparison
if len(model_metrics) >= 2 and 'unet' in model_metrics and 'reversed_ae' in model_metrics:
    unet_auc = model_metrics['unet']['auc_roc']
    ra_auc = model_metrics['reversed_ae']['auc_roc']
    auc_diff = ra_auc - unet_auc
    
    print("\\n🎯 **RESEARCH QUESTION RQ2 VALIDATION**")
    print(f"   Question: Does RA outperform U-Net?")
    print(f"   U-Net AUC:  {unet_auc:.4f}")
    print(f"   RA AUC:     {ra_auc:.4f}")
    print(f"   Difference: {auc_diff:+.4f}")
    
    # Statistical significance test
    unet_errors = model_errors['unet']
    ra_errors = model_errors['reversed_ae']
    
    # Mann-Whitney U test for error distributions
    u_stat, p_value = stats.mannwhitneyu(
        unet_errors[test_labels == 1],  # Pneumonia errors
        ra_errors[test_labels == 1],    # Pneumonia errors
        alternative='two-sided'
    )
    
    print(f"   Statistical test p-value: {p_value:.4f}")
    
    if ra_auc > unet_auc and p_value < 0.05:
        print("   ✅ **H2 CONFIRMED**: RA shows superior performance")
        print("   ✅ **RQ2 ANSWER: YES** - RA outperforms U-Net significantly")
    elif ra_auc > unet_auc:
        print("   ⚠️ **H2 PARTIAL**: RA slightly better but not statistically significant")
        print("   ⚠️ **RQ2 ANSWER: INCONCLUSIVE** - Need more data for significance")
    else:
        print("   ❌ **H2 REJECTED**: RA does not outperform U-Net")
        print("   ❌ **RQ2 ANSWER: NO** - U-Net remains superior")
        
    # Check localization quality (error separation)
    unet_separation = model_metrics['unet']['error_separation']
    ra_separation = model_metrics['reversed_ae']['error_separation']
    
    print("\\n📍 **LOCALIZATION ANALYSIS**:")
    print(f"   U-Net Error Separation: {unet_separation:.4f}")
    print(f"   RA Error Separation:    {ra_separation:.4f}")
    
    if ra_separation > unet_separation:
        print("   ✅ RA shows better error separation (localization)")
    else:
        print("   ✅ U-Net shows better error separation (localization)")

else:
    print("\\n⚠️ **RQ2 Cannot be answered**: Need both models trained")
    print("   Set CONFIG['model_type'] = 'both' to train both models")

print("=" * 80)
```

---

## **Cell 24: Comparative ROC Curves và Error Distribution**

```python
# ========================================
# COMPARATIVE VISUALIZATION
# ========================================

# Setup plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Colors for models
colors = {'unet': 'blue', 'reversed_ae': 'red'}
model_names_display = {'unet': 'U-Net', 'reversed_ae': 'Reversed AE'}

# Plot 1: Comparative ROC Curves
axes[0,0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
for model_name, metrics in model_metrics.items():
    axes[0,0].plot(
        metrics['fpr'], metrics['tpr'], 
        color=colors[model_name], linewidth=2,
        label=f"{model_names_display[model_name]} (AUC={metrics['auc_roc']:.3f})"
    )
axes[0,0].set_xlabel('False Positive Rate')
axes[0,0].set_ylabel('True Positive Rate')
axes[0,0].set_title('ROC Curves Comparison')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Comparative Precision-Recall Curves
for model_name, metrics in model_metrics.items():
    axes[0,1].plot(
        metrics['recall'], metrics['precision'],
        color=colors[model_name], linewidth=2,
        label=f"{model_names_display[model_name]} (AUC-PR={metrics['auc_pr']:.3f})"
    )
axes[0,1].set_xlabel('Recall')
axes[0,1].set_ylabel('Precision')
axes[0,1].set_title('Precision-Recall Curves')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: AUC Comparison Bar Chart
model_names = list(model_metrics.keys())
auc_scores = [model_metrics[name]['auc_roc'] for name in model_names]
display_names = [model_names_display[name] for name in model_names]

bars = axes[0,2].bar(display_names, auc_scores, 
                    color=[colors[name] for name in model_names], alpha=0.7)
axes[0,2].axhline(y=0.80, color='green', linestyle='--', linewidth=2, label='Target (0.80)')
axes[0,2].set_ylabel('AUC-ROC Score')
axes[0,2].set_title('Model Performance Comparison')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, auc_scores):
    height = bar.get_height()
    axes[0,2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                  f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Error Distribution Comparison  
for model_name, errors in model_errors.items():
    normal_errors = errors[test_labels == 0]
    pneumonia_errors = errors[test_labels == 1]
    
    axes[1,0].hist(normal_errors, bins=30, alpha=0.6, 
                  color=colors[model_name], label=f'{model_names_display[model_name]} NORMAL',
                  density=True, histtype='step', linewidth=2)
    axes[1,0].hist(pneumonia_errors, bins=30, alpha=0.6,
                  color=colors[model_name], label=f'{model_names_display[model_name]} PNEUMONIA', 
                  density=True, linestyle='--', histtype='step', linewidth=2)

axes[1,0].set_xlabel('Reconstruction Error (MSE)')
axes[1,0].set_ylabel('Density')
axes[1,0].set_title('Error Distribution by Model & Class')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 5: Box Plot Comparison
box_data = []
box_labels = []
box_colors = []

for model_name, errors in model_errors.items():
    normal_errors = errors[test_labels == 0]
    pneumonia_errors = errors[test_labels == 1]
    
    box_data.extend([normal_errors, pneumonia_errors])
    box_labels.extend([f'{model_names_display[model_name]}\\nNORMAL', 
                      f'{model_names_display[model_name]}\\nPNEUMONIA'])
    box_colors.extend([colors[model_name], colors[model_name]])

box_plot = axes[1,1].boxplot(box_data, labels=box_labels, patch_artist=True)
for patch, color in zip(box_plot['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[1,1].set_ylabel('Reconstruction Error (MSE)')
axes[1,1].set_title('Error Distribution Box Plot')
axes[1,1].grid(True, alpha=0.3)
plt.setp(axes[1,1].get_xticklabels(), rotation=45)

# Plot 6: Error Separation Analysis
model_names = list(model_metrics.keys())
separations = [model_metrics[name]['error_separation'] for name in model_names]
display_names = [model_names_display[name] for name in model_names]

bars = axes[1,2].bar(display_names, separations,
                    color=[colors[name] for name in model_names], alpha=0.7)
axes[1,2].set_ylabel('Error Separation (Pneumonia - Normal)')
axes[1,2].set_title('Anomaly Signal Strength')
axes[1,2].grid(True, alpha=0.3)

# Add value labels
for bar, sep in zip(bars, separations):
    height = bar.get_height()
    axes[1,2].text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                  f'{sep:.4f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Complete Model Comparison Analysis', fontsize=16, y=0.98)
plt.tight_layout()
plt.show()

# Save comparative results
plt.savefig(f'{CONFIG["checkpoint_dir"]}/comparative_evaluation_results.png', dpi=300, bbox_inches='tight')
print("✅ Comparative evaluation plots saved to Google Drive")
```

---

## **Cell 25: Error Heatmaps - Qualitative Analysis**

```python
# ========================================
# COMPARATIVE ERROR HEATMAPS
# ========================================

def generate_comparative_heatmaps(models, images, labels, errors_dict, num_samples=6):
    """Generate comparative error heatmaps for both models"""
    
    # Select samples: 3 normal, 3 pneumonia với different error levels
    normal_indices = np.where(labels == 0)[0]
    pneumonia_indices = np.where(labels == 1)[0]
    
    # Select diverse samples (low, medium, high error)
    normal_errors_avg = np.mean([errors_dict[model][normal_indices] for model in models.keys()], axis=0)
    pneumonia_errors_avg = np.mean([errors_dict[model][pneumonia_indices] for model in models.keys()], axis=0)
    
    normal_selected = normal_indices[np.argsort(normal_errors_avg)[[0, len(normal_errors_avg)//2, -1]]]
    pneumonia_selected = pneumonia_indices[np.argsort(pneumonia_errors_avg)[[0, len(pneumonia_errors_avg)//2, -1]]]
    
    selected_indices = np.concatenate([normal_selected, pneumonia_selected])
    
    # Create subplot grid: (2 + num_models) rows × num_samples columns
    num_models = len(models)
    fig, axes = plt.subplots(2 + num_models, len(selected_indices), figsize=(20, 4*(2 + num_models)))
    
    for i, idx in enumerate(selected_indices):
        image = images[idx:idx+1].to(device)
        class_name = "NORMAL" if labels[idx] == 0 else "PNEUMONIA"
        
        # Row 0: Original images
        orig_np = image[0, 0].cpu().numpy()
        axes[0, i].imshow(orig_np, cmap='gray', vmin=-1, vmax=1)
        axes[0, i].set_title(f'{class_name}\\nOriginal', fontsize=10)
        axes[0, i].axis('off')
        
        # Rows 1+: Each model's reconstruction and error
        for model_idx, (model_name, model) in enumerate(models.items()):
            model.eval()
            with torch.no_grad():
                reconstructed = model(image)
                error_map = torch.abs(image - reconstructed)
            
            recon_np = reconstructed[0, 0].cpu().numpy()
            error_np = error_map[0, 0].cpu().numpy()
            error_score = errors_dict[model_name][idx]
            
            # Reconstruction
            row = 1 + model_idx
            axes[row, i].imshow(recon_np, cmap='gray', vmin=-1, vmax=1)
            axes[row, i].set_title(f'{model_names_display[model_name]}\\nReconstructed\\n(Error: {error_score:.4f})', fontsize=9)
            axes[row, i].axis('off')
        
        # Last row: Error comparison overlay
        axes[-1, i].imshow(orig_np, cmap='gray', alpha=0.7, vmin=-1, vmax=1)
        
        # Overlay error maps from all models
        for model_idx, (model_name, model) in enumerate(models.items()):
            model.eval()
            with torch.no_grad():
                reconstructed = model(image)
                error_map = torch.abs(image - reconstructed)
            error_np = error_map[0, 0].cpu().numpy()
            
            # Use different alpha for each model
            alpha = 0.3 + (model_idx * 0.2)
            axes[-1, i].imshow(error_np, cmap='hot', alpha=alpha, vmin=0, vmax=error_np.max())
        
        axes[-1, i].set_title('Error Overlay', fontsize=10)
        axes[-1, i].axis('off')
    
    # Add row labels
    row_labels = ['Original'] + [model_names_display[name] for name in models.keys()] + ['Error Overlay']
    for i, label in enumerate(row_labels):
        axes[i, 0].text(-0.1, 0.5, label, transform=axes[i, 0].transAxes,
                       rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
    
    plt.suptitle('Comparative Reconstruction Error Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Save heatmaps
    plt.savefig(f'{CONFIG["checkpoint_dir"]}/comparative_error_heatmaps.png', dpi=300, bbox_inches='tight')
    print("✅ Comparative error heatmaps saved")

# Generate comparative heatmaps
if len(eval_models) > 0:
    generate_comparative_heatmaps(eval_models, test_images, test_labels, model_errors)
```

---

## **Cell 26: Final Research Report**

```python
# ========================================
# COMPREHENSIVE RESEARCH REPORT
# ========================================

import json
from datetime import datetime

# Create comprehensive research report
research_report = {
    "metadata": {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "dataset": CONFIG['hf_dataset'],
        "models_evaluated": list(model_metrics.keys()),
        "test_samples": {
            "total": len(test_labels),
            "normal": int(np.sum(test_labels == 0)),
            "pneumonia": int(np.sum(test_labels == 1))
        }
    },
    "model_performance": {},
    "research_validation": {
        "RQ1": {},
        "RQ2": {},
        "H1": {},
        "H2": {}
    },
    "statistical_analysis": {},
    "conclusions": {}
}

# Add model performance
for model_name, metrics in model_metrics.items():
    research_report["model_performance"][model_name] = {
        "auc_roc": float(metrics['auc_roc']),
        "auc_pr": float(metrics['auc_pr']),
        "f1_score": float(metrics['f1_score']),
        "sensitivity": float(metrics['sensitivity']),
        "specificity": float(metrics['specificity']),
        "error_separation": float(metrics['error_separation']),
        "optimal_threshold": float(metrics['optimal_threshold'])
    }

# Research Question 1 Validation
if 'unet' in model_metrics:
    unet_auc = model_metrics['unet']['auc_roc']
    research_report["research_validation"]["RQ1"] = {
        "question": "Can U-Net achieve effective baseline (AUC > 0.80) for pneumonia detection?",
        "measured_auc": float(unet_auc),
        "threshold": 0.80,
        "result": "CONFIRMED" if unet_auc > 0.80 else "REJECTED",
        "answer": "YES" if unet_auc > 0.80 else "NO",
        "evidence": f"U-Net achieved AUC-ROC of {unet_auc:.4f}",
        "interpretation": "Skip connections enable effective reconstruction-based anomaly detection" if unet_auc > 0.80 else "Baseline performance insufficient, may need architectural improvements"
    }
    
    research_report["research_validation"]["H1"] = {
        "hypothesis": "U-Net will achieve good baseline performance (AUC > 0.80)",
        "status": "CONFIRMED" if unet_auc > 0.80 else "REJECTED",
        "evidence": f"AUC-ROC = {unet_auc:.4f}"
    }

# Research Question 2 Validation
if len(model_metrics) >= 2 and 'unet' in model_metrics and 'reversed_ae' in model_metrics:
    unet_auc = model_metrics['unet']['auc_roc'] 
    ra_auc = model_metrics['reversed_ae']['auc_roc']
    auc_difference = ra_auc - unet_auc
    
    # Statistical test
    unet_pneumonia_errors = model_errors['unet'][test_labels == 1]
    ra_pneumonia_errors = model_errors['reversed_ae'][test_labels == 1]
    u_stat, p_value = stats.mannwhitneyu(unet_pneumonia_errors, ra_pneumonia_errors, alternative='two-sided')
    
    research_report["research_validation"]["RQ2"] = {
        "question": "Does RA demonstrate superior performance compared to U-Net?",
        "unet_auc": float(unet_auc),
        "ra_auc": float(ra_auc),
        "auc_difference": float(auc_difference),
        "statistical_test": {
            "test": "Mann-Whitney U",
            "p_value": float(p_value),
            "significant": p_value < 0.05
        },
        "result": "SUPERIOR" if ra_auc > unet_auc and p_value < 0.05 else "NOT_SUPERIOR",
        "answer": "YES" if ra_auc > unet_auc and p_value < 0.05 else "NO",
        "interpretation": "RA shows statistically significant improvement" if ra_auc > unet_auc and p_value < 0.05 else "RA does not demonstrate clear superiority"
    }
    
    # Localization analysis
    unet_separation = model_metrics['unet']['error_separation']
    ra_separation = model_metrics['reversed_ae']['error_separation']
    
    research_report["research_validation"]["H2"] = {
        "hypothesis": "RA will produce more localized error maps at pneumonia regions",
        "unet_error_separation": float(unet_separation),
        "ra_error_separation": float(ra_separation),
        "better_localization": "RA" if ra_separation > unet_separation else "U-Net",
        "status": "CONFIRMED" if ra_separation > unet_separation else "REJECTED"
    }

# Save comprehensive report
report_path = f"{CONFIG['checkpoint_dir']}/comprehensive_research_report.json"
with open(report_path, 'w') as f:
    json.dump(research_report, f, indent=2)

# Print final summary
print("\\n" + "="*100)
print("🎯 **FINAL RESEARCH VALIDATION SUMMARY**")
print("="*100)

if 'unet' in model_metrics:
    rq1_result = research_report["research_validation"]["RQ1"]
    print(f"\\n📊 **RQ1 - U-Net Baseline Effectiveness**")
    print(f"   Question: {rq1_result['question']}")
    print(f"   Answer: **{rq1_result['answer']}** (AUC: {rq1_result['measured_auc']:.4f})")
    print(f"   Status: {rq1_result['result']}")

if 'RQ2' in research_report["research_validation"] and research_report["research_validation"]["RQ2"]:
    rq2_result = research_report["research_validation"]["RQ2"]
    print(f"\\n⚖️  **RQ2 - RA vs U-Net Comparison**")
    print(f"   Question: {rq2_result['question']}")
    print(f"   Answer: **{rq2_result['answer']}**")
    print(f"   U-Net AUC: {rq2_result['unet_auc']:.4f}")
    print(f"   RA AUC: {rq2_result['ra_auc']:.4f}")
    print(f"   Difference: {rq2_result['auc_difference']:+.4f}")
    print(f"   Statistical significance: {'YES' if rq2_result['statistical_test']['significant'] else 'NO'} (p={rq2_result['statistical_test']['p_value']:.4f})")

print(f"\\n📈 **Model Performance Summary**:")
for model_name, performance in research_report["model_performance"].items():
    print(f"   {model_names_display[model_name]}:")
    print(f"     • AUC-ROC: {performance['auc_roc']:.4f}")
    print(f"     • Sensitivity: {performance['sensitivity']:.4f}")
    print(f"     • Specificity: {performance['specificity']:.4f}")
    print(f"     • Error Separation: {performance['error_separation']:.4f}")

print(f"\\n📁 **Complete report saved**: {report_path}")
print(f"📁 **All artifacts saved to**: {CONFIG['checkpoint_dir']}")

print("\\n✅ **RESEARCH COMPLETE**: All questions validated with empirical evidence!")
print("="*100)
```

---

## 📝 **Cách sử dụng:**

### 1. **Training Phase**:
```python
# Set để train cả 2 models
CONFIG['model_type'] = 'both'  # Trong cell 6
```

### 2. **Evaluation Phase**:
- Copy 4 cells trên (21-26) vào cuối notebook
- Chạy tuần tự sau khi training complete

### 3. **Results sẽ có**:
- ✅ **RQ1 Answer**: U-Net baseline có hiệu quả không?
- ✅ **RQ2 Answer**: RA có tốt hơn U-Net không?
- ✅ **H1/H2 Validation**: Hypothesis testing results
- ✅ **Statistical Analysis**: Mann-Whitney U test
- ✅ **Visual Comparison**: ROC curves, error heatmaps
- ✅ **Comprehensive Report**: JSON file với toàn bộ kết quả

### 🎯 **Expected Output**:
File `comprehensive_research_report.json` sẽ contain complete answers cho ALL research questions trong PROJECT_JOURNEY.md!