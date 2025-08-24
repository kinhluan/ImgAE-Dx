# ========================================
# ENHANCED COMPARATIVE ERROR HEATMAPS
# ========================================

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime

def generate_enhanced_comparative_heatmaps(models, images, labels, errors_dict, num_samples=8):
    """Generate enhanced comparative error heatmaps with detailed analysis"""
    
    print("🎨 Generating Enhanced Comparative Error Heatmaps...")
    print(f"   Number of models: {len(models)}")
    print(f"   Total test images: {len(images)}")
    print(f"   Sample strategy: {num_samples} images (diverse error levels)")
    
    # Analyze error distributions for intelligent sampling
    normal_indices = np.where(labels == 0)[0]
    pneumonia_indices = np.where(labels == 1)[0]
    
    print(f"   Normal samples available: {len(normal_indices)}")
    print(f"   Pneumonia samples available: {len(pneumonia_indices)}")
    
    # Calculate average errors across all models for sampling
    model_names = list(models.keys())
    
    # For each class, select samples with different error characteristics
    def select_diverse_samples(indices, errors_dict, n_samples):
        """Select samples with low, medium, high errors"""
        if len(indices) == 0:
            return []
        
        # Average errors across all models
        avg_errors = np.mean([errors_dict[model][indices] for model in model_names], axis=0)
        
        if len(avg_errors) < n_samples:
            return indices
        
        # Select samples from different error quantiles
        quantiles = np.linspace(0, 100, n_samples)
        selected = []
        
        for q in quantiles:
            percentile_value = np.percentile(avg_errors, q)
            # Find closest sample to this percentile
            closest_idx = np.argmin(np.abs(avg_errors - percentile_value))
            selected.append(indices[closest_idx])
        
        return selected
    
    # Select diverse samples
    n_per_class = num_samples // 2
    normal_selected = select_diverse_samples(normal_indices, errors_dict, n_per_class)
    pneumonia_selected = select_diverse_samples(pneumonia_indices, errors_dict, n_per_class)
    
    selected_indices = normal_selected + pneumonia_selected
    
    print(f"   Selected indices: {selected_indices}")
    
    # Print sample characteristics
    for i, idx in enumerate(selected_indices):
        class_name = "NORMAL" if labels[idx] == 0 else "PNEUMONIA"
        errors_for_sample = [errors_dict[model][idx] for model in model_names]
        avg_error = np.mean(errors_for_sample)
        print(f"   Sample {i+1}: {class_name} (idx={idx}), Avg Error={avg_error:.6f}")
    
    # Enhanced visualization setup
    num_models = len(models)
    num_rows = 3 + num_models  # Original + Models + Error Maps + Analysis
    
    fig = plt.figure(figsize=(4*len(selected_indices), 4*num_rows))
    
    # Create custom grid for better layout
    gs = fig.add_gridspec(num_rows, len(selected_indices), 
                         height_ratios=[1] + [1]*num_models + [1] + [0.5],
                         hspace=0.4, wspace=0.2)
    
    # Color schemes for different visualizations
    model_colors = {'unet': 'Blues', 'reversed_ae': 'Reds'}
    model_names_display = {'unet': 'U-Net', 'reversed_ae': 'Reversed AE'}
    
    # Process each sample
    sample_statistics = []
    
    for i, idx in enumerate(selected_indices):
        print(f"\n   Processing sample {i+1}/{len(selected_indices)}...")
        
        image = images[idx:idx+1].to(device)
        class_name = "NORMAL" if labels[idx] == 0 else "PNEUMONIA"
        
        sample_stats = {
            'index': idx,
            'class': class_name,
            'errors': {},
            'reconstructions': {},
            'error_maps': {}
        }
        
        # Row 0: Original images with enhanced annotations
        orig_np = image[0, 0].cpu().numpy()
        ax_orig = fig.add_subplot(gs[0, i])
        
        im_orig = ax_orig.imshow(orig_np, cmap='gray', vmin=-1, vmax=1)
        
        # Add colorbar for original image
        if i == 0:
            cbar_orig = plt.colorbar(im_orig, ax=ax_orig, shrink=0.8)
            cbar_orig.set_label('Pixel Intensity', rotation=270, labelpad=15)
        
        # Enhanced title with statistics
        orig_stats = f"Mean: {np.mean(orig_np):.3f}\nStd: {np.std(orig_np):.3f}"
        ax_orig.set_title(f'{class_name}\nSample {i+1} (idx={idx})\n{orig_stats}', 
                         fontsize=10, fontweight='bold')
        ax_orig.axis('off')
        
        # Add border color based on class
        border_color = 'green' if class_name == 'NORMAL' else 'red'
        rect = Rectangle((0, 0), orig_np.shape[1]-1, orig_np.shape[0]-1, 
                        linewidth=3, edgecolor=border_color, facecolor='none')
        ax_orig.add_patch(rect)
        
        # Process each model
        for model_idx, (model_name, model) in enumerate(models.items()):
            model.eval()
            with torch.no_grad():
                if CONFIG.get('mixed_precision', False):
                    with torch.cuda.amp.autocast():
                        reconstructed = model(image)
                else:
                    reconstructed = model(image)
                
                # Compute detailed error metrics
                error_map = torch.abs(image - reconstructed)
                mse_error = torch.mean((image - reconstructed)**2).item()
                mae_error = torch.mean(error_map).item()
                max_error = torch.max(error_map).item()
            
            recon_np = reconstructed[0, 0].cpu().numpy()
            error_np = error_map[0, 0].cpu().numpy()
            
            sample_stats['errors'][model_name] = mse_error
            sample_stats['reconstructions'][model_name] = recon_np
            sample_stats['error_maps'][model_name] = error_np
            
            # Row for each model: Reconstruction
            row = 1 + model_idx
            ax_recon = fig.add_subplot(gs[row, i])
            
            im_recon = ax_recon.imshow(recon_np, cmap='gray', vmin=-1, vmax=1)
            
            # Add colorbar for first column
            if i == 0:
                cbar_recon = plt.colorbar(im_recon, ax=ax_recon, shrink=0.8)
                cbar_recon.set_label('Reconstructed Intensity', rotation=270, labelpad=15)
            
            # Enhanced title with multiple metrics
            recon_quality = np.corrcoef(orig_np.flatten(), recon_np.flatten())[0,1]
            title_text = f'{model_names_display[model_name]}\nMSE: {mse_error:.6f}\n'
            title_text += f'MAE: {mae_error:.6f}\nCorr: {recon_quality:.3f}'
            
            ax_recon.set_title(title_text, fontsize=9)
            ax_recon.axis('off')
            
            # Add quality indicator border
            if mse_error < np.median(list(errors_dict[model_name])):
                border_color = 'green'  # Good reconstruction
            elif mse_error > np.percentile(list(errors_dict[model_name]), 75):
                border_color = 'red'    # Poor reconstruction  
            else:
                border_color = 'orange'  # Medium reconstruction
            
            rect = Rectangle((0, 0), recon_np.shape[1]-1, recon_np.shape[0]-1, 
                           linewidth=2, edgecolor=border_color, facecolor='none')
            ax_recon.add_patch(rect)
        
        # Row for error maps comparison
        ax_error = fig.add_subplot(gs[-2, i])
        
        # Create combined error visualization
        # Start with original image as base
        ax_error.imshow(orig_np, cmap='gray', alpha=0.5, vmin=-1, vmax=1)
        
        # Overlay error maps from all models
        max_error_overall = 0
        for model_idx, (model_name, model) in enumerate(models.items()):
            error_np = sample_stats['error_maps'][model_name]
            max_error_overall = max(max_error_overall, np.max(error_np))
        
        # Create composite error map
        composite_error = np.zeros_like(orig_np)
        for model_idx, (model_name, model) in enumerate(models.items()):
            error_np = sample_stats['error_maps'][model_name]
            composite_error += error_np / len(models)
        
        # Show composite error with hot colormap
        im_error = ax_error.imshow(composite_error, cmap='hot', alpha=0.7, 
                                 vmin=0, vmax=max_error_overall)
        
        if i == 0:
            cbar_error = plt.colorbar(im_error, ax=ax_error, shrink=0.8)
            cbar_error.set_label('Error Magnitude', rotation=270, labelpad=15)
        
        error_stats = f"Max: {np.max(composite_error):.4f}\nMean: {np.mean(composite_error):.4f}"
        ax_error.set_title(f'Composite Error Map\n{error_stats}', fontsize=10)
        ax_error.axis('off')
        
        # Row for detailed analysis
        ax_analysis = fig.add_subplot(gs[-1, i])
        ax_analysis.axis('off')
        
        # Create analysis text
        analysis_text = f"ANALYSIS SAMPLE {i+1}\n" + "="*20 + "\n"
        analysis_text += f"Class: {class_name}\n"
        analysis_text += f"Index: {idx}\n\n"
        
        # Model comparison
        model_errors = [sample_stats['errors'][model] for model in model_names]
        best_model_idx = np.argmin(model_errors)
        best_model = model_names[best_model_idx]
        
        analysis_text += "Model Performance:\n"
        for model_name in model_names:
            error = sample_stats['errors'][model_name]
            indicator = "⭐" if model_name == best_model else "  "
            analysis_text += f"{indicator} {model_names_display[model_name]}: {error:.6f}\n"
        
        # Error difference
        if len(model_names) == 2:
            error_diff = abs(model_errors[0] - model_errors[1])
            analysis_text += f"\nError Difference: {error_diff:.6f}\n"
            
            if error_diff < 0.001:
                analysis_text += "Similar performance\n"
            elif error_diff < 0.01:
                analysis_text += "Moderate difference\n"
            else:
                analysis_text += "Significant difference\n"
        
        ax_analysis.text(0.1, 0.9, analysis_text, transform=ax_analysis.transAxes,
                        fontsize=8, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        sample_statistics.append(sample_stats)
    
    # Add overall row labels
    row_labels = ['Original'] + [model_names_display[name] for name in model_names] + ['Error Maps', 'Analysis']
    
    # Add row labels on the left
    for i, label in enumerate(row_labels):
        if i < len(row_labels) - 1:  # Skip analysis row
            ax = fig.add_subplot(gs[i, 0])
            ax.text(-0.15, 0.5, label, transform=ax.transAxes,
                   rotation=90, va='center', ha='center', 
                   fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Enhanced main title with statistics
    total_samples = len(selected_indices)
    normal_count = sum(1 for idx in selected_indices if labels[idx] == 0)
    pneumonia_count = total_samples - normal_count
    
    main_title = f"Enhanced Comparative Reconstruction Error Analysis\n"
    main_title += f"{total_samples} Samples: {normal_count} Normal, {pneumonia_count} Pneumonia | "
    main_title += f"{len(models)} Models Compared"
    
    plt.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    # Save enhanced heatmaps
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    heatmap_filename = f'{CONFIG["checkpoint_dir"]}/enhanced_error_heatmaps_{timestamp}.png'
    plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ Enhanced error heatmaps saved: {heatmap_filename}")
    
    # Generate summary statistics
    print(f"\n📊 HEATMAP ANALYSIS SUMMARY:")
    print("-" * 60)
    
    normal_samples = [s for s in sample_statistics if s['class'] == 'NORMAL']
    pneumonia_samples = [s for s in sample_statistics if s['class'] == 'PNEUMONIA']
    
    for model_name in model_names:
        print(f"\n🤖 {model_names_display[model_name].upper()}:")
        
        # Normal samples analysis
        normal_errors = [s['errors'][model_name] for s in normal_samples]
        if normal_errors:
            print(f"   NORMAL samples ({len(normal_errors)}):")
            print(f"      Mean error: {np.mean(normal_errors):.6f}")
            print(f"      Std error:  {np.std(normal_errors):.6f}")
            print(f"      Min error:  {np.min(normal_errors):.6f}")
            print(f"      Max error:  {np.max(normal_errors):.6f}")
        
        # Pneumonia samples analysis
        pneumonia_errors = [s['errors'][model_name] for s in pneumonia_samples]
        if pneumonia_errors:
            print(f"   PNEUMONIA samples ({len(pneumonia_errors)}):")
            print(f"      Mean error: {np.mean(pneumonia_errors):.6f}")
            print(f"      Std error:  {np.std(pneumonia_errors):.6f}")
            print(f"      Min error:  {np.min(pneumonia_errors):.6f}")
            print(f"      Max error:  {np.max(pneumonia_errors):.6f}")
        
        # Class separation
        if normal_errors and pneumonia_errors:
            separation = np.mean(pneumonia_errors) - np.mean(normal_errors)
            print(f"   Class Separation: {separation:.6f}")
    
    # Model comparison for each sample
    print(f"\n🏆 SAMPLE-BY-SAMPLE COMPARISON:")
    for i, stats in enumerate(sample_statistics):
        print(f"\n   Sample {i+1} ({stats['class']}):")
        model_errors = [(model, stats['errors'][model]) for model in model_names]
        model_errors.sort(key=lambda x: x[1])  # Sort by error
        
        best_model, best_error = model_errors[0]
        print(f"      Best: {model_names_display[best_model]} ({best_error:.6f})")
        
        if len(model_errors) > 1:
            worst_model, worst_error = model_errors[-1]
            difference = worst_error - best_error
            print(f"      Worst: {model_names_display[worst_model]} ({worst_error:.6f})")
            print(f"      Difference: {difference:.6f} ({(difference/best_error*100):.1f}%)")
    
    # Save detailed statistics
    stats_filename = f'{CONFIG["checkpoint_dir"]}/heatmap_statistics_{timestamp}.json'
    detailed_stats = {
        'timestamp': timestamp,
        'config': CONFIG,
        'sample_statistics': [],
        'summary': {
            'total_samples': len(sample_statistics),
            'normal_samples': len(normal_samples),
            'pneumonia_samples': len(pneumonia_samples),
            'models_compared': list(model_names)
        }
    }
    
    # Convert numpy arrays to lists for JSON serialization
    for stats in sample_statistics:
        json_stats = {
            'index': int(stats['index']),
            'class': stats['class'],
            'errors': {k: float(v) for k, v in stats['errors'].items()}
        }
        detailed_stats['sample_statistics'].append(json_stats)
    
    with open(stats_filename, 'w') as f:
        json.dump(detailed_stats, f, indent=2)
    
    print(f"\n💾 Detailed statistics saved: {stats_filename}")
    
    return sample_statistics

# ========================================
# ADDITIONAL ANALYSIS: ERROR LOCALIZATION
# ========================================

def analyze_error_localization(models, images, labels, errors_dict, num_samples=4):
    """Analyze where errors occur in images (localization analysis)"""
    
    print("\n🎯 Error Localization Analysis...")
    
    # Select samples with highest errors for each class
    normal_indices = np.where(labels == 0)[0]
    pneumonia_indices = np.where(labels == 1)[0]
    
    # Get highest error samples for detailed localization analysis
    selected_samples = []
    for class_indices, class_name in [(normal_indices, 'NORMAL'), (pneumonia_indices, 'PNEUMONIA')]:
        if len(class_indices) == 0:
            continue
            
        # For each model, find samples with highest errors
        for model_name in models.keys():
            model_errors = errors_dict[model_name][class_indices]
            highest_error_idx = class_indices[np.argmax(model_errors)]
            
            selected_samples.append({
                'index': highest_error_idx,
                'class': class_name,
                'model': model_name,
                'error': np.max(model_errors)
            })
    
    # Create localization visualization
    fig, axes = plt.subplots(len(selected_samples), 4, figsize=(16, 4*len(selected_samples)))
    if len(selected_samples) == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx, sample_info in enumerate(selected_samples):
        idx = sample_info['index']
        model_name = sample_info['model']
        model = models[model_name]
        
        image = images[idx:idx+1].to(device)
        
        model.eval()
        with torch.no_grad():
            reconstructed = model(image)
            error_map = torch.abs(image - reconstructed)
        
        orig_np = image[0, 0].cpu().numpy()
        recon_np = reconstructed[0, 0].cpu().numpy()
        error_np = error_map[0, 0].cpu().numpy()
        
        # Original
        axes[sample_idx, 0].imshow(orig_np, cmap='gray')
        axes[sample_idx, 0].set_title(f'Original\n{sample_info["class"]}')
        axes[sample_idx, 0].axis('off')
        
        # Reconstructed
        axes[sample_idx, 1].imshow(recon_np, cmap='gray')
        axes[sample_idx, 1].set_title(f'Reconstructed\n{model_names_display[model_name]}')
        axes[sample_idx, 1].axis('off')
        
        # Error map
        im_error = axes[sample_idx, 2].imshow(error_np, cmap='hot')
        axes[sample_idx, 2].set_title(f'Error Map\n(Max: {np.max(error_np):.4f})')
        axes[sample_idx, 2].axis('off')
        plt.colorbar(im_error, ax=axes[sample_idx, 2], shrink=0.8)
        
        # Error overlay
        axes[sample_idx, 3].imshow(orig_np, cmap='gray', alpha=0.7)
        axes[sample_idx, 3].imshow(error_np, cmap='hot', alpha=0.5)
        axes[sample_idx, 3].set_title('Error Overlay')
        axes[sample_idx, 3].axis('off')
    
    plt.suptitle('Error Localization Analysis - Highest Error Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Save localization analysis
    localization_filename = f'{CONFIG["checkpoint_dir"]}/error_localization_analysis_{timestamp}.png'
    plt.savefig(localization_filename, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ Error localization analysis saved: {localization_filename}")

# Execute enhanced analysis
if 'eval_models' in globals() and len(eval_models) > 0 and 'test_images' in globals():
    print("="*80)
    print("🎨 STARTING ENHANCED ERROR HEATMAP ANALYSIS")
    print("="*80)
    
    # Generate enhanced comparative heatmaps
    sample_stats = generate_enhanced_comparative_heatmaps(
        eval_models, test_images, test_labels, model_errors, num_samples=8
    )
    
    # Generate error localization analysis
    analyze_error_localization(
        eval_models, test_images, test_labels, model_errors, num_samples=4
    )
    
    # Force sync to Google Drive
    import os
    if os.path.exists('/content/drive'):
        os.system('sync')
        print("\n🔄 All visualizations synced to Google Drive")
    
    print("\n🎉 Enhanced heatmap analysis completed!")
    
else:
    print("❌ Required data not available for heatmap analysis")
    print("   Need: eval_models, test_images, test_labels, model_errors")
    print("   Please run model evaluation first")