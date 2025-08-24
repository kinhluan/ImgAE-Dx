# ========================================
# ENHANCED COMPARATIVE ANALYSIS & VISUALIZATION
# ========================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    f1_score, confusion_matrix, classification_report
)
from datetime import datetime
import json
import seaborn as sns

# First, compute metrics for all models
print("="*80)
print("🎯 ENHANCED COMPARATIVE ANALYSIS")
print("="*80)

def compute_comprehensive_metrics(errors, labels, model_name):
    """Compute comprehensive anomaly detection metrics"""
    print(f"\n📊 Computing metrics for {model_name}...")
    
    # Basic statistics
    normal_errors = errors[labels == 0]
    pneumonia_errors = errors[labels == 1]
    
    print(f"   Normal samples: {len(normal_errors)}")
    print(f"   Pneumonia samples: {len(pneumonia_errors)}")
    
    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(labels, errors)
    auc_roc = roc_auc_score(labels, errors)
    
    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(labels, errors)
    auc_pr = np.trapz(precision, recall)
    
    # Optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = roc_thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]
    
    # Predictions at optimal threshold
    predictions = (errors >= optimal_threshold).astype(int)
    
    # F1 Score
    f1 = f1_score(labels, predictions)
    
    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Error separation analysis
    normal_mean = np.mean(normal_errors)
    pneumonia_mean = np.mean(pneumonia_errors)
    error_separation = pneumonia_mean - normal_mean
    separation_ratio = pneumonia_mean / normal_mean if normal_mean > 0 else float('inf')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(normal_errors) - 1) * np.var(normal_errors) + 
                         (len(pneumonia_errors) - 1) * np.var(pneumonia_errors)) /
                        (len(normal_errors) + len(pneumonia_errors) - 2))
    cohens_d = error_separation / pooled_std if pooled_std > 0 else 0
    
    # Overlap analysis
    normal_max = np.max(normal_errors)
    pneumonia_min = np.min(pneumonia_errors)
    overlap_range = max(0, normal_max - pneumonia_min)
    
    metrics = {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'f1_score': f1,
        'accuracy': accuracy,
        'precision': precision_score,
        'recall': recall_score,
        'sensitivity': optimal_sensitivity,
        'specificity': optimal_specificity,
        'optimal_threshold': optimal_threshold,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'pr_precision': precision,
        'pr_recall': recall,
        'pr_thresholds': pr_thresholds,
        'confusion_matrix': cm,
        'normal_mean_error': normal_mean,
        'pneumonia_mean_error': pneumonia_mean,
        'error_separation': error_separation,
        'separation_ratio': separation_ratio,
        'cohens_d': cohens_d,
        'normal_std': np.std(normal_errors),
        'pneumonia_std': np.std(pneumonia_errors),
        'overlap_range': overlap_range,
        'normal_errors': normal_errors,
        'pneumonia_errors': pneumonia_errors
    }
    
    # Print detailed results
    print(f"   🎯 Performance Metrics:")
    print(f"      AUC-ROC: {auc_roc:.4f}")
    print(f"      AUC-PR:  {auc_pr:.4f}")
    print(f"      F1-Score: {f1:.4f}")
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      Sensitivity: {optimal_sensitivity:.4f}")
    print(f"      Specificity: {optimal_specificity:.4f}")
    print(f"      Optimal Threshold: {optimal_threshold:.6f}")
    
    print(f"   📈 Error Analysis:")
    print(f"      Normal Mean Error: {normal_mean:.6f} ± {np.std(normal_errors):.6f}")
    print(f"      Pneumonia Mean Error: {pneumonia_mean:.6f} ± {np.std(pneumonia_errors):.6f}")
    print(f"      Error Separation: {error_separation:.6f}")
    print(f"      Separation Ratio: {separation_ratio:.2f}x")
    print(f"      Cohen's d (effect size): {cohens_d:.3f}")
    print(f"      Overlap Range: {overlap_range:.6f}")
    
    # Research question evaluation
    rq1_passed = auc_roc >= 0.80
    print(f"   🔬 Research Question 1:")
    print(f"      Target AUC ≥ 0.80: {'✅ PASSED' if rq1_passed else '❌ FAILED'} ({auc_roc:.4f})")
    
    return metrics

# Check if required data exists
if 'model_errors' not in globals() or 'test_labels' not in globals():
    print("❌ Required data not available. Please run model evaluation first.")
    print("   Need: model_errors, test_labels")
else:
    # Compute metrics for all models
    model_metrics = {}
    
    for model_name, errors in model_errors.items():
        model_metrics[model_name] = compute_comprehensive_metrics(
            errors, test_labels, model_name.upper()
        )

    print(f"\n🏆 COMPARATIVE RESULTS SUMMARY:")
    print("-" * 60)
    
    # Create comparison table
    comparison_data = []
    for model_name, metrics in model_metrics.items():
        comparison_data.append([
            model_name.upper(),
            f"{metrics['auc_roc']:.4f}",
            f"{metrics['auc_pr']:.4f}",
            f"{metrics['f1_score']:.4f}",
            f"{metrics['accuracy']:.4f}",
            f"{metrics['error_separation']:.6f}",
            f"{metrics['separation_ratio']:.2f}x",
            f"{metrics['cohens_d']:.3f}",
            "✅" if metrics['auc_roc'] >= 0.80 else "❌"
        ])
    
    headers = ["Model", "AUC-ROC", "AUC-PR", "F1", "Accuracy", "Error Sep", "Sep Ratio", "Cohen's d", "RQ1"]
    
    print(f"{'Model':<12} {'AUC-ROC':<8} {'AUC-PR':<8} {'F1':<8} {'Acc':<8} {'ErrorSep':<10} {'SepRatio':<9} {'Cohen_d':<8} {'RQ1':<4}")
    print("-" * 80)
    for data in comparison_data:
        print(f"{data[0]:<12} {data[1]:<8} {data[2]:<8} {data[3]:<8} {data[4]:<8} {data[5]:<10} {data[6]:<9} {data[7]:<8} {data[8]:<4}")

    # ========================================
    # ENHANCED COMPARATIVE VISUALIZATION
    # ========================================
    
    print(f"\n📊 Creating enhanced comparative visualizations...")
    
    # Setup enhanced plot layout
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Colors and display names
    colors = {'unet': '#2E86C1', 'reversed_ae': '#E74C3C'}
    model_names_display = {'unet': 'U-Net', 'reversed_ae': 'Reversed AE'}
    
    # Plot 1: ROC Curves with confidence bands
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random Classifier')
    
    for model_name, metrics in model_metrics.items():
        ax1.plot(
            metrics['fpr'], metrics['tpr'],
            color=colors[model_name], linewidth=3,
            label=f"{model_names_display[model_name]} (AUC={metrics['auc_roc']:.3f})"
        )
        # Mark optimal point
        optimal_idx = np.argmax(metrics['tpr'] - metrics['fpr'])
        ax1.plot(metrics['fpr'][optimal_idx], metrics['tpr'][optimal_idx], 
                'o', color=colors[model_name], markersize=8, markerfacecolor='white', markeredgewidth=2)
    
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Plot 2: Precision-Recall Curves
    ax2 = fig.add_subplot(gs[0, 1])
    
    for model_name, metrics in model_metrics.items():
        ax2.plot(
            metrics['pr_recall'], metrics['pr_precision'],
            color=colors[model_name], linewidth=3,
            label=f"{model_names_display[model_name]} (AUC-PR={metrics['auc_pr']:.3f})"
        )
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    # Plot 3: Performance Metrics Radar Chart
    ax3 = fig.add_subplot(gs[0, 2], projection='polar')
    
    metrics_names = ['AUC-ROC', 'AUC-PR', 'F1-Score', 'Sensitivity', 'Specificity']
    angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for model_name, metrics in model_metrics.items():
        values = [
            metrics['auc_roc'],
            metrics['auc_pr'], 
            metrics['f1_score'],
            metrics['sensitivity'],
            metrics['specificity']
        ]
        values += values[:1]  # Complete the circle
        
        ax3.plot(angles, values, 'o-', linewidth=2, color=colors[model_name],
                label=model_names_display[model_name])
        ax3.fill(angles, values, alpha=0.25, color=colors[model_name])
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(metrics_names, fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax3.grid(True)
    
    # Plot 4: AUC Comparison with Target Line
    ax4 = fig.add_subplot(gs[0, 3])
    
    model_names = list(model_metrics.keys())
    auc_scores = [model_metrics[name]['auc_roc'] for name in model_names]
    display_names = [model_names_display[name] for name in model_names]
    
    bars = ax4.bar(display_names, auc_scores,
                   color=[colors[name] for name in model_names], 
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add target line
    ax4.axhline(y=0.80, color='green', linestyle='--', linewidth=2, 
               label='Research Target (0.80)', alpha=0.8)
    
    # Add value labels on bars
    for bar, score in zip(bars, auc_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Add pass/fail indicator
        status = "✅" if score >= 0.80 else "❌"
        ax4.text(bar.get_x() + bar.get_width()/2., height/2,
                status, ha='center', va='center', fontsize=16)
    
    ax4.set_ylabel('AUC-ROC Score', fontsize=12)
    ax4.set_title('AUC-ROC Comparison vs Target', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1])
    
    # Plot 5: Error Distribution Overlays
    ax5 = fig.add_subplot(gs[1, 0:2])
    
    x_min = float('inf')
    x_max = float('-inf')
    
    for model_name, errors in model_errors.items():
        normal_errors = errors[test_labels == 0]
        pneumonia_errors = errors[test_labels == 1]
        
        x_min = min(x_min, np.min(errors))
        x_max = max(x_max, np.max(errors))
        
        # Create histograms
        bins = np.linspace(x_min, x_max, 50)
        
        ax5.hist(normal_errors, bins=bins, alpha=0.6, color=colors[model_name], 
                label=f'{model_names_display[model_name]} - NORMAL (n={len(normal_errors)})',
                density=True, histtype='step', linewidth=2)
        ax5.hist(pneumonia_errors, bins=bins, alpha=0.6, color=colors[model_name],
                label=f'{model_names_display[model_name]} - PNEUMONIA (n={len(pneumonia_errors)})',
                density=True, histtype='stepfilled', linestyle='--', linewidth=2)
        
        # Add vertical lines for means
        ax5.axvline(np.mean(normal_errors), color=colors[model_name], 
                   linestyle='-', alpha=0.8, linewidth=1)
        ax5.axvline(np.mean(pneumonia_errors), color=colors[model_name], 
                   linestyle='--', alpha=0.8, linewidth=1)
    
    ax5.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
    ax5.set_ylabel('Density', fontsize=12)
    ax5.set_title('Error Distribution Comparison by Class', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Box Plot with Statistical Tests
    ax6 = fig.add_subplot(gs[1, 2])
    
    box_data = []
    box_labels = []
    box_colors = []
    positions = []
    pos = 1
    
    for model_name, errors in model_errors.items():
        normal_errors = errors[test_labels == 0]
        pneumonia_errors = errors[test_labels == 1]
        
        box_data.extend([normal_errors, pneumonia_errors])
        box_labels.extend([f'{model_names_display[model_name]}\nNORMAL', 
                          f'{model_names_display[model_name]}\nPNEUMONIA'])
        box_colors.extend([colors[model_name], colors[model_name]])
        positions.extend([pos, pos+0.4])
        pos += 1
    
    box_plot = ax6.boxplot(box_data, positions=positions, patch_artist=True,
                          labels=box_labels, widths=0.35)
    
    for patch, color in zip(box_plot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    
    ax6.set_ylabel('Reconstruction Error (MSE)', fontsize=12)
    ax6.set_title('Error Distribution Box Plot', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 7: Error Separation Analysis
    ax7 = fig.add_subplot(gs[1, 3])
    
    model_names = list(model_metrics.keys())
    separations = [model_metrics[name]['error_separation'] for name in model_names]
    separation_ratios = [model_metrics[name]['separation_ratio'] for name in model_names]
    display_names = [model_names_display[name] for name in model_names]
    
    bars = ax7.bar(display_names, separations,
                   color=[colors[name] for name in model_names], alpha=0.8,
                   edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, sep, ratio in zip(bars, separations, separation_ratios):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{sep:.4f}\n({ratio:.1f}x)', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    ax7.set_ylabel('Error Separation\n(Pneumonia - Normal)', fontsize=12)
    ax7.set_title('Anomaly Signal Strength', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Plot 8: Confusion Matrices
    ax8 = fig.add_subplot(gs[2, 0])
    ax9 = fig.add_subplot(gs[2, 1])
    
    axes_cm = [ax8, ax9]
    model_list = list(model_metrics.keys())
    
    for idx, (model_name, metrics) in enumerate(model_metrics.items()):
        if idx < len(axes_cm):
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Pneumonia'],
                       yticklabels=['Normal', 'Pneumonia'],
                       ax=axes_cm[idx])
            axes_cm[idx].set_title(f'{model_names_display[model_name]} Confusion Matrix', 
                                  fontsize=12, fontweight='bold')
            axes_cm[idx].set_xlabel('Predicted', fontsize=11)
            axes_cm[idx].set_ylabel('Actual', fontsize=11)
    
    # Plot 9: Research Questions Summary
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.axis('off')
    
    # Create research questions summary text
    summary_text = "RESEARCH QUESTIONS VALIDATION\n" + "="*50 + "\n\n"
    
    summary_text += "🎯 RQ1: Can U-Net achieve AUC > 0.80 for anomaly detection?\n"
    for model_name, metrics in model_metrics.items():
        auc = metrics['auc_roc']
        status = "✅ PASSED" if auc >= 0.80 else "❌ FAILED"
        summary_text += f"   {model_names_display[model_name]}: {status} (AUC = {auc:.4f})\n"
    
    summary_text += "\n🔬 RQ2: Does RA show better localization than U-Net?\n"
    if len(model_metrics) >= 2:
        models = list(model_metrics.keys())
        unet_cohens_d = model_metrics.get('unet', {}).get('cohens_d', 0)
        ra_cohens_d = model_metrics.get('reversed_ae', {}).get('cohens_d', 0)
        
        if ra_cohens_d > unet_cohens_d:
            summary_text += f"   ✅ RA shows stronger effect size (Cohen's d = {ra_cohens_d:.3f} vs {unet_cohens_d:.3f})\n"
        else:
            summary_text += f"   ⚠️ U-Net shows stronger effect size (Cohen's d = {unet_cohens_d:.3f} vs {ra_cohens_d:.3f})\n"
    
    summary_text += "\n📊 STATISTICAL SUMMARY:\n"
    for model_name, metrics in model_metrics.items():
        summary_text += f"\n🤖 {model_names_display[model_name].upper()}:\n"
        summary_text += f"   Performance: AUC-ROC={metrics['auc_roc']:.4f}, F1={metrics['f1_score']:.4f}\n"
        summary_text += f"   Threshold: {metrics['optimal_threshold']:.6f}\n"
        summary_text += f"   Sensitivity/Specificity: {metrics['sensitivity']:.3f}/{metrics['specificity']:.3f}\n"
        summary_text += f"   Error Separation: {metrics['error_separation']:.6f} ({metrics['separation_ratio']:.1f}x)\n"
        summary_text += f"   Effect Size (Cohen's d): {metrics['cohens_d']:.3f}\n"
    
    summary_text += f"\n💡 CONCLUSION:\n"
    best_model = max(model_metrics.items(), key=lambda x: x[1]['auc_roc'])
    summary_text += f"   Best performing model: {model_names_display[best_model[0]]} (AUC={best_model[1]['auc_roc']:.4f})\n"
    
    research_success = any(metrics['auc_roc'] >= 0.80 for metrics in model_metrics.values())
    summary_text += f"   Research objectives: {'✅ ACHIEVED' if research_success else '❌ NOT ACHIEVED'}\n"
    
    ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Main title
    plt.suptitle('Complete Comparative Analysis - Medical Image Anomaly Detection', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f'{CONFIG["checkpoint_dir"]}/comparative_analysis_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Save detailed results
    results_filename = f'{CONFIG["checkpoint_dir"]}/comparative_results_{timestamp}.json'
    
    # Prepare results for JSON (convert numpy arrays)
    json_results = {
        'timestamp': timestamp,
        'config': CONFIG,
        'model_metrics': {}
    }
    
    for model_name, metrics in model_metrics.items():
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                json_metrics[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                json_metrics[key] = float(value)
            else:
                json_metrics[key] = value
        json_results['model_metrics'][model_name] = json_metrics
    
    with open(results_filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Enhanced comparative analysis completed!")
    print(f"📊 Visualization saved: {plot_filename}")
    print(f"💾 Detailed results saved: {results_filename}")
    print(f"📁 All files in: {CONFIG['checkpoint_dir']}")
    
    # Force sync to Google Drive
    import os
    if os.path.exists('/content/drive'):
        os.system('sync')
        print("🔄 Files synced to Google Drive")

print("="*80)