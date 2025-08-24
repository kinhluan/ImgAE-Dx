# ========================================
# MULTI-IMAGE ENHANCED COMPARATIVE HEATMAPS
# ========================================

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime
import math

def generate_multi_image_heatmaps(models, images, labels, errors_dict, num_samples=20):
    """Generate enhanced comparative heatmaps with many more samples"""
    
    print(f"🎨 Generating Multi-Image Comparative Error Heatmaps...")
    print(f"   Number of models: {len(models)}")
    print(f"   Total test images: {len(images)}")
    print(f"   Requested samples: {num_samples}")
    
    # Analyze error distributions for intelligent sampling
    normal_indices = np.where(labels == 0)[0]
    pneumonia_indices = np.where(labels == 1)[0]
    
    print(f"   Normal samples available: {len(normal_indices)}")
    print(f"   Pneumonia samples available: {len(pneumonia_indices)}")
    
    model_names = list(models.keys())
    model_names_display = {'unet': 'U-Net', 'reversed_ae': 'Reversed AE'}
    
    # Enhanced sample selection strategy
    def select_comprehensive_samples(indices, errors_dict, n_samples, class_name):
        """Select samples across full error spectrum"""
        if len(indices) == 0:
            return []
        
        # Calculate average errors across all models
        avg_errors = np.mean([errors_dict[model][indices] for model in model_names], axis=0)
        
        # Also calculate per-model errors for variety
        model_specific_errors = {model: errors_dict[model][indices] for model in model_names}
        
        selected = []
        
        # Strategy 1: Error percentiles (50% of samples)
        percentile_samples = n_samples // 2
        if percentile_samples > 0:
            percentiles = np.linspace(0, 100, percentile_samples)
            for p in percentiles:
                percentile_value = np.percentile(avg_errors, p)
                closest_idx = np.argmin(np.abs(avg_errors - percentile_value))
                if indices[closest_idx] not in [s for s in selected]:
                    selected.append(indices[closest_idx])
        
        # Strategy 2: Model-specific high errors (25% of samples)
        model_specific_samples = n_samples // 4
        for model in model_names:
            model_errors = model_specific_errors[model]
            # Get highest errors for this model
            high_error_indices = np.argsort(model_errors)[-model_specific_samples:]
            for idx in high_error_indices:
                if indices[idx] not in selected and len(selected) < n_samples:
                    selected.append(indices[idx])
        
        # Strategy 3: Random sampling from remaining (fill up to n_samples)
        remaining_indices = [idx for idx in indices if idx not in selected]
        if len(remaining_indices) > 0 and len(selected) < n_samples:
            np.random.seed(42)  # Reproducible
            additional_needed = min(n_samples - len(selected), len(remaining_indices))
            additional = np.random.choice(remaining_indices, additional_needed, replace=False)
            selected.extend(additional)
        
        print(f"   {class_name} samples selected: {len(selected)}")
        return selected[:n_samples]  # Ensure we don't exceed limit
    
    # Select samples for each class
    samples_per_class = num_samples // 2
    normal_selected = select_comprehensive_samples(normal_indices, errors_dict, samples_per_class, "NORMAL")
    pneumonia_selected = select_comprehensive_samples(pneumonia_indices, errors_dict, samples_per_class, "PNEUMONIA")
    
    # Combine and organize samples
    selected_indices = normal_selected + pneumonia_selected
    actual_samples = len(selected_indices)
    
    print(f"   Final sample count: {actual_samples}")
    print(f"   Normal: {len(normal_selected)}, Pneumonia: {len(pneumonia_selected)}")
    
    # Calculate optimal layout
    # For many samples, use multiple figures or scrollable layout
    max_cols = 10  # Maximum columns per figure
    samples_per_figure = 20  # Maximum samples per figure
    
    if actual_samples <= samples_per_figure:
        # Single figure
        figures_needed = 1
        samples_in_figures = [actual_samples]
    else:
        # Multiple figures
        figures_needed = math.ceil(actual_samples / samples_per_figure)
        samples_in_figures = []
        for i in range(figures_needed):
            start_idx = i * samples_per_figure
            end_idx = min((i + 1) * samples_per_figure, actual_samples)
            samples_in_figures.append(end_idx - start_idx)
    
    print(f"   Creating {figures_needed} figure(s)")
    
    all_sample_statistics = []
    
    # Generate figures
    for fig_idx in range(figures_needed):
        print(f"\n   📊 Generating Figure {fig_idx + 1}/{figures_needed}")
        
        # Calculate sample range for this figure
        start_sample = fig_idx * samples_per_figure
        end_sample = start_sample + samples_in_figures[fig_idx]
        figure_samples = selected_indices[start_sample:end_sample]
        
        # Calculate layout for this figure
        n_samples_fig = len(figure_samples)
        cols = min(max_cols, n_samples_fig)
        
        # Enhanced row structure
        num_models = len(models)
        rows_needed = 2 + num_models + 2  # Original + Models + Error + Stats
        
        # Create figure with appropriate size
        fig_width = max(20, 2.5 * cols)  # Minimum 20, scale with columns
        fig_height = max(15, 3 * rows_needed)  # Scale with rows
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Custom grid layout
        gs = fig.add_gridspec(rows_needed, cols,
                             height_ratios=[1.2] + [1]*num_models + [1, 0.8],
                             hspace=0.4, wspace=0.15)
        
        figure_statistics = []
        
        # Process each sample in this figure
        for col_idx, idx in enumerate(figure_samples):
            sample_idx_global = start_sample + col_idx
            print(f"     Processing sample {sample_idx_global + 1}/{actual_samples} (idx={idx})")
            
            image = images[idx:idx+1].to(device)
            class_name = "NORMAL" if labels[idx] == 0 else "PNEUMONIA"
            
            sample_stats = {
                'figure': fig_idx + 1,
                'position': col_idx,
                'global_index': sample_idx_global,
                'data_index': idx,
                'class': class_name,
                'errors': {},
                'metrics': {}
            }
            
            # Row 0: Original images with enhanced info
            orig_np = image[0, 0].cpu().numpy()
            ax_orig = fig.add_subplot(gs[0, col_idx])
            
            im_orig = ax_orig.imshow(orig_np, cmap='gray', vmin=-1, vmax=1)
            
            # Enhanced original image statistics
            orig_mean = np.mean(orig_np)
            orig_std = np.std(orig_np)
            orig_min = np.min(orig_np)
            orig_max = np.max(orig_np)
            
            title_text = f'{class_name}\\n#{sample_idx_global + 1} (idx={idx})'
            title_text += f'\\nμ={orig_mean:.2f}, σ={orig_std:.2f}'
            title_text += f'\\n[{orig_min:.2f}, {orig_max:.2f}]'
            
            ax_orig.set_title(title_text, fontsize=8, fontweight='bold')
            ax_orig.axis('off')
            
            # Add class-based border
            border_color = '#2E8B57' if class_name == 'NORMAL' else '#DC143C'  # Sea green / Crimson
            rect = Rectangle((0, 0), orig_np.shape[1]-1, orig_np.shape[0]-1,
                           linewidth=2, edgecolor=border_color, facecolor='none', alpha=0.8)
            ax_orig.add_patch(rect)
            
            # Process each model
            model_errors_for_sample = {}
            model_reconstructions = {}
            
            for model_idx, (model_name, model) in enumerate(models.items()):
                model.eval()
                with torch.no_grad():
                    if CONFIG.get('mixed_precision', False):
                        with torch.cuda.amp.autocast():
                            reconstructed = model(image)
                    else:
                        reconstructed = model(image)
                    
                    # Comprehensive error metrics
                    error_map = torch.abs(image - reconstructed)
                    squared_error_map = (image - reconstructed)**2
                    
                    mse_error = torch.mean(squared_error_map).item()
                    mae_error = torch.mean(error_map).item()
                    max_error = torch.max(error_map).item()
                    rmse_error = torch.sqrt(torch.mean(squared_error_map)).item()
                
                recon_np = reconstructed[0, 0].cpu().numpy()
                error_np = error_map[0, 0].cpu().numpy()
                
                # Calculate additional metrics
                correlation = np.corrcoef(orig_np.flatten(), recon_np.flatten())[0,1]
                ssim_approx = 1 - (2 * np.mean((orig_np - recon_np)**2)) / (np.var(orig_np) + np.var(recon_np) + (np.mean(orig_np) - np.mean(recon_np))**2)
                
                model_errors_for_sample[model_name] = mse_error
                model_reconstructions[model_name] = {
                    'image': recon_np,
                    'error_map': error_np,
                    'mse': mse_error,
                    'mae': mae_error,
                    'max_error': max_error,
                    'rmse': rmse_error,
                    'correlation': correlation,
                    'ssim_approx': ssim_approx
                }
                
                # Row for each model reconstruction
                row = 1 + model_idx
                ax_recon = fig.add_subplot(gs[row, col_idx])
                
                im_recon = ax_recon.imshow(recon_np, cmap='gray', vmin=-1, vmax=1)
                
                # Comprehensive title with multiple metrics
                model_display_name = model_names_display.get(model_name, model_name)
                title_text = f'{model_display_name}'
                title_text += f'\\nMSE: {mse_error:.4f}'
                title_text += f'\\nMAE: {mae_error:.4f}'
                title_text += f'\\nCorr: {correlation:.3f}'
                title_text += f'\\nSSIM: {ssim_approx:.3f}'
                
                ax_recon.set_title(title_text, fontsize=7)
                ax_recon.axis('off')
                
                # Quality-based border color
                # Use percentiles of all errors for this model to determine quality
                all_errors_this_model = list(errors_dict[model_name])
                percentile_25 = np.percentile(all_errors_this_model, 25)
                percentile_75 = np.percentile(all_errors_this_model, 75)
                
                if mse_error <= percentile_25:
                    quality_color = '#228B22'  # Forest green - excellent
                elif mse_error <= percentile_75:
                    quality_color = '#FFA500'  # Orange - average
                else:
                    quality_color = '#FF4500'  # Orange red - poor
                
                rect = Rectangle((0, 0), recon_np.shape[1]-1, recon_np.shape[0]-1,
                               linewidth=2, edgecolor=quality_color, facecolor='none', alpha=0.8)
                ax_recon.add_patch(rect)
            
            # Row: Composite error visualization
            ax_error = fig.add_subplot(gs[-2, col_idx])
            
            # Create sophisticated error overlay
            ax_error.imshow(orig_np, cmap='gray', alpha=0.6, vmin=-1, vmax=1)
            
            # Combine error maps from all models
            combined_errors = []
            max_error_overall = 0
            
            for model_name in model_names:
                error_map = model_reconstructions[model_name]['error_map']
                combined_errors.append(error_map)
                max_error_overall = max(max_error_overall, np.max(error_map))
            
            # Average error map
            avg_error_map = np.mean(combined_errors, axis=0)
            
            # Show error with hot colormap
            im_error = ax_error.imshow(avg_error_map, cmap='hot', alpha=0.7,
                                     vmin=0, vmax=max_error_overall)
            
            # Error statistics
            error_mean = np.mean(avg_error_map)
            error_max = np.max(avg_error_map)
            error_std = np.std(avg_error_map)
            
            title_text = f'Avg Error Map'
            title_text += f'\\nMax: {error_max:.4f}'
            title_text += f'\\nMean: {error_mean:.4f}'
            title_text += f'\\nStd: {error_std:.4f}'
            
            ax_error.set_title(title_text, fontsize=7)
            ax_error.axis('off')
            
            # Row: Detailed statistics and comparison
            ax_stats = fig.add_subplot(gs[-1, col_idx])
            ax_stats.axis('off')
            
            # Determine best and worst models
            model_performance = [(model, model_errors_for_sample[model]) for model in model_names]
            model_performance.sort(key=lambda x: x[1])
            
            best_model, best_error = model_performance[0]
            worst_model, worst_error = model_performance[-1] if len(model_performance) > 1 else model_performance[0]
            
            # Create detailed statistics text
            stats_text = f"ANALYSIS\\n" + "="*12 + "\\n"
            stats_text += f"Class: {class_name}\\n"
            stats_text += f"Sample: {sample_idx_global + 1}\\n\\n"
            
            stats_text += "Performance:\\n"
            for i, (model, error) in enumerate(model_performance):
                indicator = "🥇" if i == 0 else "🥈" if i == 1 and len(model_performance) > 1 else "  "
                model_display = model_names_display.get(model, model)
                stats_text += f"{indicator} {model_display}:\\n"
                stats_text += f"   {error:.5f}\\n"
            
            # Performance difference
            if len(model_performance) > 1:
                error_diff = worst_error - best_error
                relative_diff = (error_diff / best_error) * 100
                stats_text += f"\\nDifference:\\n"
                stats_text += f"{error_diff:.5f}\\n"
                stats_text += f"({relative_diff:.1f}%)\\n"
                
                if relative_diff < 5:
                    stats_text += "Similar\\n"
                elif relative_diff < 20:
                    stats_text += "Moderate\\n"
                else:
                    stats_text += "Significant\\n"
            
            # Additional metrics for best model
            best_metrics = model_reconstructions[best_model]
            stats_text += f"\\nBest Model:\\n"
            stats_text += f"RMSE: {best_metrics['rmse']:.4f}\\n"
            stats_text += f"Corr: {best_metrics['correlation']:.3f}\\n"
            
            ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                         fontsize=6, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.9))
            
            # Store statistics
            sample_stats['errors'] = model_errors_for_sample
            sample_stats['metrics'] = {model: model_reconstructions[model] for model in model_names}
            sample_stats['best_model'] = best_model
            sample_stats['worst_model'] = worst_model
            sample_stats['performance_difference'] = worst_error - best_error if len(model_performance) > 1 else 0
            
            figure_statistics.append(sample_stats)
            all_sample_statistics.extend([sample_stats])
        
        # Add row labels for this figure
        row_labels = ['Original'] + [model_names_display.get(name, name) for name in model_names] + ['Error Maps', 'Statistics']
        
        for row_idx, label in enumerate(row_labels):
            if row_idx < len(row_labels):
                # Add label on the left side
                ax_label = fig.add_subplot(gs[row_idx, 0])
                ax_label.text(-0.2, 0.5, label, transform=ax_label.transAxes,
                             rotation=90, va='center', ha='center',
                             fontsize=12, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # Enhanced figure title
        normal_count_fig = sum(1 for s in figure_statistics if s['class'] == 'NORMAL')
        pneumonia_count_fig = len(figure_statistics) - normal_count_fig
        
        fig_title = f"Multi-Image Comparative Analysis - Figure {fig_idx + 1}/{figures_needed}\\n"
        fig_title += f"{len(figure_statistics)} Samples: {normal_count_fig} Normal, {pneumonia_count_fig} Pneumonia | "
        fig_title += f"Samples {start_sample + 1}-{end_sample} of {actual_samples}"
        
        plt.suptitle(fig_title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
        
        # Save each figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_filename = f'{CONFIG["checkpoint_dir"]}/multi_image_heatmaps_fig{fig_idx + 1}_{timestamp}.png'
        plt.savefig(fig_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"     ✅ Figure {fig_idx + 1} saved: {fig_filename}")
    
    # Generate comprehensive summary
    print(f"\\n📊 COMPREHENSIVE MULTI-IMAGE ANALYSIS SUMMARY:")
    print("="*80)
    
    # Overall statistics
    total_normal = sum(1 for s in all_sample_statistics if s['class'] == 'NORMAL')
    total_pneumonia = len(all_sample_statistics) - total_normal
    
    print(f"Total Samples Analyzed: {len(all_sample_statistics)}")
    print(f"   Normal: {total_normal}")
    print(f"   Pneumonia: {total_pneumonia}")
    print(f"   Figures Generated: {figures_needed}")
    
    # Model performance summary
    for model_name in model_names:
        model_display = model_names_display.get(model_name, model_name)
        print(f"\\n🤖 {model_display.upper()} PERFORMANCE:")
        
        # All errors for this model
        all_errors = [s['errors'][model_name] for s in all_sample_statistics]
        normal_errors = [s['errors'][model_name] for s in all_sample_statistics if s['class'] == 'NORMAL']
        pneumonia_errors = [s['errors'][model_name] for s in all_sample_statistics if s['class'] == 'PNEUMONIA']
        
        print(f"   Overall: Mean={np.mean(all_errors):.6f}, Std={np.std(all_errors):.6f}")
        if normal_errors:
            print(f"   Normal:  Mean={np.mean(normal_errors):.6f}, Std={np.std(normal_errors):.6f}")
        if pneumonia_errors:
            print(f"   Pneumonia: Mean={np.mean(pneumonia_errors):.6f}, Std={np.std(pneumonia_errors):.6f}")
        
        # Best performance count
        best_count = sum(1 for s in all_sample_statistics if s['best_model'] == model_name)
        print(f"   Best Model on: {best_count}/{len(all_sample_statistics)} samples ({best_count/len(all_sample_statistics)*100:.1f}%)")
    
    # Class-wise model preference
    normal_samples = [s for s in all_sample_statistics if s['class'] == 'NORMAL']
    pneumonia_samples = [s for s in all_sample_statistics if s['class'] == 'PNEUMONIA']
    
    if normal_samples:
        print(f"\\n📊 NORMAL SAMPLES MODEL PREFERENCE:")
        for model_name in model_names:
            model_display = model_names_display.get(model_name, model_name)
            count = sum(1 for s in normal_samples if s['best_model'] == model_name)
            print(f"   {model_display}: {count}/{len(normal_samples)} ({count/len(normal_samples)*100:.1f}%)")
    
    if pneumonia_samples:
        print(f"\\n📊 PNEUMONIA SAMPLES MODEL PREFERENCE:")
        for model_name in model_names:
            model_display = model_names_display.get(model_name, model_name)
            count = sum(1 for s in pneumonia_samples if s['best_model'] == model_name)
            print(f"   {model_display}: {count}/{len(pneumonia_samples)} ({count/len(pneumonia_samples)*100:.1f}%)")
    
    # Performance difference analysis
    performance_diffs = [s['performance_difference'] for s in all_sample_statistics if s['performance_difference'] > 0]
    if performance_diffs:
        print(f"\\n📈 PERFORMANCE DIFFERENCES:")
        print(f"   Mean difference: {np.mean(performance_diffs):.6f}")
        print(f"   Max difference: {np.max(performance_diffs):.6f}")
        print(f"   Significant differences (>20%): {sum(1 for s in all_sample_statistics if s['performance_difference']/min(s['errors'].values()) > 0.2 if s['performance_difference'] > 0)}")
    
    # Save comprehensive statistics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comprehensive_stats_file = f'{CONFIG["checkpoint_dir"]}/multi_image_comprehensive_stats_{timestamp}.json'
    
    # Prepare data for JSON
    json_stats = {
        'timestamp': timestamp,
        'config': CONFIG,
        'analysis_summary': {
            'total_samples': len(all_sample_statistics),
            'normal_samples': total_normal,
            'pneumonia_samples': total_pneumonia,
            'figures_generated': figures_needed,
            'models_compared': list(model_names)
        },
        'sample_statistics': []
    }
    
    for stats in all_sample_statistics:
        json_sample = {
            'figure': stats['figure'],
            'position': stats['position'],
            'global_index': stats['global_index'],
            'data_index': int(stats['data_index']),
            'class': stats['class'],
            'errors': {k: float(v) for k, v in stats['errors'].items()},
            'best_model': stats['best_model'],
            'worst_model': stats['worst_model'],
            'performance_difference': float(stats['performance_difference'])
        }
        json_stats['sample_statistics'].append(json_sample)
    
    with open(comprehensive_stats_file, 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    print(f"\\n💾 Comprehensive statistics saved: {comprehensive_stats_file}")
    
    return all_sample_statistics

# Execute multi-image analysis
if 'eval_models' in globals() and len(eval_models) > 0 and 'test_images' in globals():
    print("="*80)
    print("🎨 STARTING MULTI-IMAGE HEATMAP ANALYSIS")
    print("="*80)
    
    # Generate multi-image heatmaps with many samples
    multi_stats = generate_multi_image_heatmaps(
        eval_models, test_images, test_labels, model_errors, 
        num_samples=30  # Increase to 30 samples for comprehensive analysis
    )
    
    # Force sync to Google Drive
    import os
    if os.path.exists('/content/drive'):
        os.system('sync')
        print("\\n🔄 All multi-image visualizations synced to Google Drive")
    
    print("\\n🎉 Multi-image heatmap analysis completed!")
    print(f"   Generated visualizations for {len(multi_stats)} samples")
    
else:
    print("❌ Required data not available for multi-image heatmap analysis")
    print("   Need: eval_models, test_images, test_labels, model_errors")
    print("   Please run model evaluation first")