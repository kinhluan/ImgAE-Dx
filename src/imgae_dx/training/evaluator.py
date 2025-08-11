"""
Model evaluator for comprehensive performance assessment.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..models.base import BaseAutoencoder
from .metrics import AnomalyMetrics


class Evaluator:
    """
    Comprehensive model evaluator for medical image anomaly detection.
    
    Provides detailed analysis including ROC curves, error maps, 
    score distributions, and model comparisons.
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or self._get_device()
        self.metrics = AnomalyMetrics()
        self.results_cache = {}
        
    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def evaluate_single_model(
        self,
        model: BaseAutoencoder,
        test_loader: DataLoader,
        normal_label: int = 0,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on test data.
        
        Args:
            model: Trained autoencoder model
            test_loader: Test data loader
            normal_label: Label for normal samples
            model_name: Name for identification
            
        Returns:
            Comprehensive evaluation results
        """
        model_name = model_name or model.model_name
        model.to(self.device)
        model.eval()
        
        print(f"Evaluating {model_name}...")
        
        # Collect predictions and targets
        all_scores = []
        all_labels = []
        all_reconstructions = []
        all_originals = []
        latent_representations = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                images = images.to(self.device)
                
                # Get reconstructions and latent representations
                if hasattr(model, 'forward_with_latent'):
                    reconstructions, latents = model.forward_with_latent(images)
                    latent_representations.append(latents.cpu())
                else:
                    reconstructions = model(images)
                    latents = model.encode(images)
                    latent_representations.append(latents.cpu())
                
                # Compute anomaly scores
                scores = self.metrics.compute_anomaly_scores(
                    images, reconstructions, method='mse'
                )
                
                # Store results
                all_scores.extend(scores)
                all_labels.extend(labels.numpy())
                all_reconstructions.append(reconstructions.cpu())
                all_originals.append(images.cpu())
        
        # Convert to arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        binary_labels = (all_labels != normal_label).astype(int)
        
        # Concatenate tensors
        all_reconstructions = torch.cat(all_reconstructions, dim=0)
        all_originals = torch.cat(all_originals, dim=0)
        latent_representations = torch.cat(latent_representations, dim=0)
        
        # Compute metrics
        metrics_results = self.metrics.evaluate_anomaly_detection(
            binary_labels, all_scores
        )
        
        # Additional analysis
        normal_scores = all_scores[binary_labels == 0]
        anomaly_scores = all_scores[binary_labels == 1]
        
        results = {
            'model_name': model_name,
            'scores': all_scores,
            'labels': binary_labels,
            'normal_scores': normal_scores,
            'anomaly_scores': anomaly_scores,
            'reconstructions': all_reconstructions,
            'originals': all_originals,
            'latents': latent_representations,
            'metrics': metrics_results,
            'score_stats': {
                'normal_mean': float(np.mean(normal_scores)),
                'normal_std': float(np.std(normal_scores)),
                'anomaly_mean': float(np.mean(anomaly_scores)),
                'anomaly_std': float(np.std(anomaly_scores)),
                'separation': float(np.mean(anomaly_scores) - np.mean(normal_scores))
            }
        }
        
        # Cache results
        self.results_cache[model_name] = results
        
        print(f"{model_name} Results:")
        print(f"  AUC-ROC: {metrics_results['auc_roc']:.4f}")
        print(f"  AUC-PR:  {metrics_results['auc_pr']:.4f}")
        print(f"  F1:      {metrics_results['f1_score']:.4f}")
        print(f"  Score separation: {results['score_stats']['separation']:.4f}")
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, BaseAutoencoder],
        test_loader: DataLoader,
        normal_label: int = 0,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same test set.
        
        Args:
            models: Dictionary of {name: model}
            test_loader: Test data loader
            normal_label: Label for normal samples
            save_dir: Directory to save plots
            
        Returns:
            Comparison results
        """
        print(f"Comparing {len(models)} models...")
        
        # Evaluate each model
        model_results = {}
        for name, model in models.items():
            results = self.evaluate_single_model(
                model, test_loader, normal_label, name
            )
            model_results[name] = results
        
        # Extract metrics for comparison
        comparison_metrics = {}
        for name, results in model_results.items():
            comparison_metrics[name] = results['metrics']
        
        # Create comparison plots
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Model comparison chart
            self.metrics.compare_models(
                comparison_metrics,
                save_path=str(save_dir / "model_comparison.png")
            )
            
            # ROC curves comparison
            self.plot_roc_comparison(
                model_results,
                save_path=str(save_dir / "roc_comparison.png")
            )
            
            # Score distributions comparison
            self.plot_score_distributions_comparison(
                model_results,
                save_path=str(save_dir / "score_distributions.png")
            )
        
        # Determine best model
        best_model = max(
            comparison_metrics.items(),
            key=lambda x: x[1]['auc_roc']
        )
        
        comparison_results = {
            'individual_results': model_results,
            'comparison_metrics': comparison_metrics,
            'best_model': {
                'name': best_model[0],
                'auc_roc': best_model[1]['auc_roc']
            },
            'ranking': sorted(
                comparison_metrics.items(),
                key=lambda x: x[1]['auc_roc'],
                reverse=True
            )
        }
        
        print("\nModel Ranking (by AUC-ROC):")
        for i, (name, metrics) in enumerate(comparison_results['ranking'], 1):
            print(f"  {i}. {name}: {metrics['auc_roc']:.4f}")
        
        return comparison_results
    
    def plot_roc_comparison(
        self,
        model_results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot ROC curves for model comparison."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name, results in model_results.items():
            scores = results['scores']
            labels = results['labels']
            auc = results['metrics']['auc_roc']
            
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(labels, scores)
            
            ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_score_distributions_comparison(
        self,
        model_results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot score distributions for all models."""
        n_models = len(model_results)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (name, results) in enumerate(model_results.items()):
            ax = axes[i]
            
            normal_scores = results['normal_scores']
            anomaly_scores = results['anomaly_scores']
            
            ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', 
                   density=True, color='blue')
            ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', 
                   density=True, color='red')
            
            ax.axvline(np.mean(normal_scores), color='blue', linestyle='--')
            ax.axvline(np.mean(anomaly_scores), color='red', linestyle='--')
            
            ax.set_title(f'{name} - Score Distribution')
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_reconstructions(
        self,
        model_results: Dict[str, Dict],
        n_samples: int = 8,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize reconstructions from different models."""
        n_models = len(model_results)
        fig, axes = plt.subplots(3, n_samples, figsize=(2 * n_samples, 6))
        
        # Get random samples
        first_model = list(model_results.values())[0]
        indices = np.random.choice(len(first_model['originals']), n_samples, replace=False)
        
        for j in range(n_samples):
            idx = indices[j]
            
            # Original image
            original = first_model['originals'][idx].squeeze().numpy()
            axes[0, j].imshow(original, cmap='gray')
            axes[0, j].set_title('Original' if j == 0 else '')
            axes[0, j].axis('off')
            
            # Model reconstructions (show first two models)
            model_names = list(model_results.keys())[:2]
            
            for i, model_name in enumerate(model_names):
                reconstruction = model_results[model_name]['reconstructions'][idx].squeeze().numpy()
                axes[i + 1, j].imshow(reconstruction, cmap='gray')
                axes[i + 1, j].set_title(f'{model_name}' if j == 0 else '')
                axes[i + 1, j].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_error_maps(
        self,
        model_results: Dict[str, Dict],
        n_samples: int = 4,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Generate error maps showing reconstruction differences."""
        n_models = len(model_results)
        fig, axes = plt.subplots(n_models, n_samples, figsize=(3 * n_samples, 3 * n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        # Get samples with highest anomaly scores
        first_model = list(model_results.values())[0]
        top_anomaly_indices = np.argsort(first_model['scores'])[-n_samples:]
        
        for i, (model_name, results) in enumerate(model_results.items()):
            for j, idx in enumerate(top_anomaly_indices):
                original = results['originals'][idx].squeeze().numpy()
                reconstruction = results['reconstructions'][idx].squeeze().numpy()
                
                # Compute error map
                error_map = np.abs(original - reconstruction)
                
                im = axes[i, j].imshow(error_map, cmap='hot', interpolation='nearest')
                axes[i, j].set_title(f'{model_name}\nScore: {results["scores"][idx]:.3f}')
                axes[i, j].axis('off')
                
                # Add colorbar to last column
                if j == n_samples - 1:
                    plt.colorbar(im, ax=axes[i, j])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_latent_space(
        self,
        model_results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze latent space representations."""
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        analysis_results = {}
        
        for model_name, results in model_results.items():
            latents = results['latents'].numpy()
            labels = results['labels']
            
            # PCA analysis
            pca = PCA(n_components=2)
            latents_pca = pca.fit_transform(latents)
            
            # t-SNE analysis
            tsne = TSNE(n_components=2, random_state=42)
            latents_tsne = tsne.fit_transform(latents[:1000])  # Subset for speed
            
            analysis_results[model_name] = {
                'pca_components': latents_pca,
                'tsne_components': latents_tsne,
                'pca_explained_variance': pca.explained_variance_ratio_,
                'labels': labels
            }
        
        # Plot latent space visualizations
        if save_path:
            self.plot_latent_space_analysis(analysis_results, save_path)
        
        return analysis_results
    
    def plot_latent_space_analysis(
        self,
        analysis_results: Dict[str, Dict],
        save_path: str
    ):
        """Plot latent space analysis results."""
        n_models = len(analysis_results)
        fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (model_name, results) in enumerate(analysis_results.items()):
            # PCA plot
            pca_components = results['pca_components']
            labels = results['labels']
            
            scatter = axes[0, i].scatter(
                pca_components[:, 0], pca_components[:, 1],
                c=labels, cmap='viridis', alpha=0.6
            )
            axes[0, i].set_title(f'{model_name} - PCA\n'
                               f'Var Explained: {results["pca_explained_variance"].sum():.3f}')
            axes[0, i].set_xlabel('PC1')
            axes[0, i].set_ylabel('PC2')
            plt.colorbar(scatter, ax=axes[0, i])
            
            # t-SNE plot
            tsne_components = results['tsne_components']
            tsne_labels = labels[:1000]
            
            scatter = axes[1, i].scatter(
                tsne_components[:, 0], tsne_components[:, 1],
                c=tsne_labels, cmap='viridis', alpha=0.6
            )
            axes[1, i].set_title(f'{model_name} - t-SNE')
            axes[1, i].set_xlabel('t-SNE 1')
            axes[1, i].set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=axes[1, i])
        
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def generate_comprehensive_report(
        self,
        comparison_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """Generate a comprehensive evaluation report."""
        report_lines = [
            "# ImgAE-Dx Model Evaluation Report",
            "=" * 50,
            "",
            f"**Evaluation Date**: {torch.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Models Evaluated**: {len(comparison_results['individual_results'])}",
            "",
            "## Model Performance Summary",
            "-" * 30
        ]
        
        # Add model rankings
        for i, (name, metrics) in enumerate(comparison_results['ranking'], 1):
            report_lines.extend([
                f"",
                f"### {i}. {name}",
                f"- **AUC-ROC**: {metrics['auc_roc']:.4f}",
                f"- **AUC-PR**: {metrics['auc_pr']:.4f}",
                f"- **F1-Score**: {metrics['f1_score']:.4f}",
                f"- **Accuracy**: {metrics['accuracy']:.4f}",
                f"- **Sensitivity**: {metrics['sensitivity']:.4f}",
                f"- **Specificity**: {metrics['specificity']:.4f}"
            ])
        
        # Add detailed analysis for best model
        best_model_name = comparison_results['best_model']['name']
        best_results = comparison_results['individual_results'][best_model_name]
        
        report_lines.extend([
            "",
            "## Best Model Detailed Analysis",
            "-" * 35,
            f"**Model**: {best_model_name}",
            f"**AUC-ROC**: {best_results['metrics']['auc_roc']:.4f}",
            "",
            "### Score Statistics:",
            f"- Normal samples mean score: {best_results['score_stats']['normal_mean']:.4f} ± {best_results['score_stats']['normal_std']:.4f}",
            f"- Anomaly samples mean score: {best_results['score_stats']['anomaly_mean']:.4f} ± {best_results['score_stats']['anomaly_std']:.4f}",
            f"- Score separation: {best_results['score_stats']['separation']:.4f}",
            ""
        ])
        
        # Add recommendations
        best_auc = comparison_results['best_model']['auc_roc']
        if best_auc > 0.9:
            performance = "Excellent"
            recommendation = "Model is ready for clinical evaluation."
        elif best_auc > 0.8:
            performance = "Good"
            recommendation = "Consider fine-tuning or ensemble methods."
        elif best_auc > 0.7:
            performance = "Fair"
            recommendation = "Significant improvements needed before deployment."
        else:
            performance = "Poor"
            recommendation = "Model requires major architecture changes."
        
        report_lines.extend([
            "## Recommendations",
            "-" * 20,
            f"**Overall Performance**: {performance}",
            f"**Recommendation**: {recommendation}",
            "",
            "## Next Steps",
            "-" * 15,
            "1. Analyze error cases for model improvement insights",
            "2. Consider ensemble methods combining multiple models",
            "3. Evaluate on additional validation datasets",
            "4. Conduct clinical validation if performance is satisfactory"
        ])
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report