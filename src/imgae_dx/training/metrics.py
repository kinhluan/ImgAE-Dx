"""
Evaluation metrics for medical image anomaly detection.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns


class AnomalyMetrics:
    """
    Comprehensive metrics for anomaly detection evaluation.
    
    Specifically designed for medical imaging anomaly detection tasks.
    """
    
    def __init__(self):
        self.results_history = []
    
    def compute_reconstruction_error(
        self, 
        original: torch.Tensor, 
        reconstruction: torch.Tensor,
        reduction: str = 'none'
    ) -> torch.Tensor:
        """
        Compute reconstruction error between original and reconstructed images.
        
        Args:
            original: Original images (N, C, H, W)
            reconstruction: Reconstructed images (N, C, H, W)
            reduction: 'none', 'mean', 'sum', or 'sample_mean'
            
        Returns:
            Reconstruction error tensor
        """
        # Compute per-pixel squared error
        error = torch.pow(original - reconstruction, 2)
        
        if reduction == 'none':
            return error
        elif reduction == 'mean':
            return torch.mean(error)
        elif reduction == 'sum':
            return torch.sum(error)
        elif reduction == 'sample_mean':
            # Mean error per sample
            return torch.mean(error, dim=(1, 2, 3))
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    def compute_anomaly_scores(
        self,
        original: torch.Tensor,
        reconstruction: torch.Tensor,
        method: str = 'mse'
    ) -> np.ndarray:
        """
        Compute anomaly scores for each sample.
        
        Args:
            original: Original images
            reconstruction: Reconstructed images
            method: Scoring method ('mse', 'mae', 'ssim')
            
        Returns:
            Anomaly scores array
        """
        if method == 'mse':
            scores = self.compute_reconstruction_error(
                original, reconstruction, reduction='sample_mean'
            )
        elif method == 'mae':
            scores = torch.mean(torch.abs(original - reconstruction), dim=(1, 2, 3))
        elif method == 'ssim':
            scores = self._compute_ssim_scores(original, reconstruction)
        else:
            raise ValueError(f"Unknown scoring method: {method}")
        
        return scores.detach().cpu().numpy()
    
    def _compute_ssim_scores(
        self, 
        original: torch.Tensor, 
        reconstruction: torch.Tensor
    ) -> torch.Tensor:
        """Compute SSIM-based anomaly scores."""
        # Simple SSIM approximation (for full SSIM, use pytorch_ssim)
        # Here we use a simplified version focusing on luminance and contrast
        
        def ssim_single(x, y):
            mu_x = torch.mean(x)
            mu_y = torch.mean(y)
            
            sigma_x = torch.var(x)
            sigma_y = torch.var(y)
            sigma_xy = torch.mean((x - mu_x) * (y - mu_y))
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
            
            return 1 - ssim  # Convert to dissimilarity score
        
        scores = []
        for i in range(original.shape[0]):
            score = ssim_single(original[i], reconstruction[i])
            scores.append(score)
        
        return torch.stack(scores)
    
    def evaluate_anomaly_detection(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_scores: Union[np.ndarray, List[float]],
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Comprehensive anomaly detection evaluation.
        
        Args:
            y_true: True binary labels (0: normal, 1: anomaly)
            y_scores: Anomaly scores
            threshold: Classification threshold (if None, use optimal)
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # ROC metrics
        try:
            auc_roc = roc_auc_score(y_true, y_scores)
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        except ValueError:
            # Handle case where all labels are the same
            auc_roc = 0.5
            fpr = tpr = roc_thresholds = np.array([0, 1])
        
        # Precision-Recall metrics
        try:
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
            auc_pr = average_precision_score(y_true, y_scores)
        except ValueError:
            precision = recall = pr_thresholds = np.array([1, 0])
            auc_pr = np.mean(y_true)
        
        # Find optimal threshold if not provided
        if threshold is None:
            # Use Youden's J statistic for optimal threshold
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            threshold = roc_thresholds[optimal_idx]
        
        # Binary classification metrics
        y_pred = (y_scores >= threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision_score * sensitivity) / (precision_score + sensitivity) if (precision_score + sensitivity) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        results = {
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr),
            'threshold': float(threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision_score),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        # Store results
        self.results_history.append(results)
        
        return results
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot ROC curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot Precision-Recall curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auc_pr = average_precision_score(y_true, y_scores)
        
        ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {auc_pr:.3f})')
        ax.axhline(y=np.mean(y_true), color='k', linestyle='--', linewidth=1, 
                   label=f'Baseline (Prevalence = {np.mean(y_true):.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_score_distribution(
        self,
        normal_scores: np.ndarray,
        anomaly_scores: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot distribution of anomaly scores."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True, color='blue')
        ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', density=True, color='red')
        
        # Plot means
        ax.axvline(np.mean(normal_scores), color='blue', linestyle='--', 
                   label=f'Normal Mean: {np.mean(normal_scores):.3f}')
        ax.axvline(np.mean(anomaly_scores), color='red', linestyle='--',
                   label=f'Anomaly Mean: {np.mean(anomaly_scores):.3f}')
        
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Anomaly Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compare_models(
        self,
        results_dict: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Compare multiple models' performance."""
        metrics = ['auc_roc', 'auc_pr', 'f1_score', 'accuracy', 'sensitivity', 'specificity']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            models = list(results_dict.keys())
            values = [results_dict[model].get(metric, 0) for model in models]
            
            bars = ax.bar(models, values, alpha=0.7)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Model Performance Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_summary_report(self, results: Dict[str, float]) -> str:
        """Generate a summary report."""
        report = f"""
Anomaly Detection Evaluation Report
==================================

ROC Metrics:
- AUC-ROC: {results['auc_roc']:.4f}
- AUC-PR:  {results['auc_pr']:.4f}

Classification Metrics (Threshold: {results['threshold']:.4f}):
- Accuracy:    {results['accuracy']:.4f}
- Precision:   {results['precision']:.4f}
- Sensitivity: {results['sensitivity']:.4f}
- Specificity: {results['specificity']:.4f}
- F1-Score:    {results['f1_score']:.4f}

Confusion Matrix:
                Predicted
                Normal  Anomaly
Actual Normal   {results['true_negatives']:6}  {results['false_positives']:6}
       Anomaly  {results['false_negatives']:6}  {results['true_positives']:6}

Performance Interpretation:
- AUC-ROC > 0.9: Excellent
- AUC-ROC > 0.8: Good  
- AUC-ROC > 0.7: Fair
- AUC-ROC < 0.7: Poor

Current Performance: {'Excellent' if results['auc_roc'] > 0.9 else 'Good' if results['auc_roc'] > 0.8 else 'Fair' if results['auc_roc'] > 0.7 else 'Poor'}
        """
        
        return report.strip()
    
    def save_results(self, results: Dict[str, float], filepath: str):
        """Save results to JSON file."""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_results(self, filepath: str) -> Dict[str, float]:
        """Load results from JSON file."""
        import json
        
        with open(filepath, 'r') as f:
            return json.load(f)