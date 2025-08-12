"""
Base trainer class for medical image anomaly detection models.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, Tuple, List
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Mixed precision training support
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

from ..models.base import BaseAutoencoder
from ..utils import ConfigManager
from .metrics import AnomalyMetrics


class Trainer:
    """
    Base trainer for autoencoder models.
    
    Supports training, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: BaseAutoencoder,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        wandb_project: Optional[str] = None,
        save_artifacts: bool = True,
        use_mixed_precision: bool = True
    ):
        self.model = model
        self.config = config or {}
        self.device = device or self._get_device()
        self.wandb_project = wandb_project
        self.save_artifacts = save_artifacts
        
        # T4 GPU optimizations
        self.use_mixed_precision = use_mixed_precision and AMP_AVAILABLE and self.device == "cuda"
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize components
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.metrics = AnomalyMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # T4 GPU memory optimization
        self._setup_t4_optimizations()
        
        # Setup logging
        self._setup_logging()
        
    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _setup_t4_optimizations(self):
        """Setup T4 GPU specific optimizations."""
        if self.device == "cuda":
            # Enable optimizations for T4 GPU
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Check GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"🔧 T4 GPU Optimizations:")
            print(f"   Mixed Precision: {'✅ Enabled' if self.use_mixed_precision else '❌ Disabled'}")
            print(f"   GPU Memory: {gpu_memory_gb:.1f} GB")
            print(f"   cuDNN Benchmark: ✅ Enabled")
            
            # Set memory fraction for T4 (leave some memory for system)
            if gpu_memory_gb <= 16:  # T4 GPU detection
                torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of T4's 16GB
                print(f"   Memory Fraction: 85% (T4 optimized)")
    
    def get_optimal_batch_size(self, base_batch_size: int = 32) -> int:
        """
        Calculate optimal batch size for T4 GPU based on available memory.
        
        Args:
            base_batch_size: Starting batch size
            
        Returns:
            Optimized batch size for T4
        """
        if self.device != "cuda":
            return base_batch_size
            
        try:
            # Check available GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            # T4-specific batch size recommendations
            if gpu_memory_gb <= 16:  # T4 GPU
                # Conservative batch sizes for medical images (128x128)
                if self.use_mixed_precision:
                    # Mixed precision allows larger batches
                    recommended_batch_size = min(64, base_batch_size * 2)
                else:
                    # Full precision - smaller batches
                    recommended_batch_size = min(32, base_batch_size)
            else:
                # Other GPUs
                recommended_batch_size = base_batch_size
                
            print(f"🎯 Batch Size Optimization:")
            print(f"   Base: {base_batch_size} → Optimized: {recommended_batch_size}")
            
            return recommended_batch_size
            
        except Exception as e:
            print(f"⚠️  Could not optimize batch size: {e}")
            return base_batch_size
    
    def _setup_logging(self):
        """Setup W&B logging if available."""
        try:
            import wandb
            
            if self.wandb_project and not wandb.run:
                wandb.init(
                    project=self.wandb_project,
                    config=self.config,
                    name=f"{self.model.model_name}_{int(time.time())}"
                )
                self.use_wandb = True
            else:
                self.use_wandb = False
        except ImportError:
            self.use_wandb = False
    
    def setup_training(
        self,
        learning_rate: float = 1e-4,
        optimizer_name: str = "adam",
        loss_fn: Optional[nn.Module] = None,
        scheduler_name: Optional[str] = None,
        **kwargs
    ):
        """Setup training components."""
        
        # Setup optimizer
        if optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=kwargs.get('weight_decay', 1e-5)
            )
        elif optimizer_name.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=kwargs.get('weight_decay', 1e-5)
            )
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=kwargs.get('weight_decay', 1e-5)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Setup loss function
        if loss_fn is None:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = loss_fn
        
        # Setup scheduler
        if scheduler_name:
            if scheduler_name.lower() == "cosine":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=kwargs.get('T_max', 100)
                )
            elif scheduler_name.lower() == "step":
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=kwargs.get('step_size', 30),
                    gamma=kwargs.get('gamma', 0.1)
                )
            elif scheduler_name.lower() == "plateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=kwargs.get('factor', 0.5),
                    patience=kwargs.get('patience', 10)
                )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with mixed precision support."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)  # T4 optimization
            
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                # Mixed precision training for T4 GPU
                with autocast():
                    reconstruction = self.model(images)
                    loss = self.loss_fn(reconstruction, images)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                reconstruction = self.model(images)
                loss = self.loss_fn(reconstruction, images)
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            batch_loss = loss.item()
            epoch_loss += batch_loss
            
            # T4 memory management
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()  # Clear cache every 10 batches
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                'mixed_prec': '✅' if self.use_mixed_precision else '❌'
            })
            
            # Log to W&B
            if self.use_wandb:
                import wandb
                wandb.log({
                    'batch_loss': batch_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'mixed_precision': self.use_mixed_precision
                })
        
        avg_loss = epoch_loss / num_batches
        return {'loss': avg_loss}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch with mixed precision support."""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with autocast():
                        reconstruction = self.model(images)
                        loss = self.loss_fn(reconstruction, images)
                else:
                    reconstruction = self.model(images)
                    loss = self.loss_fn(reconstruction, images)
                
                epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_batches
        return {'loss': avg_loss}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        save_every: int = 5,
        checkpoint_dir: str = "checkpoints",
        early_stopping_patience: int = 10,
        min_delta: float = 1e-4
    ) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
            min_delta: Minimum change for early stopping
            
        Returns:
            Dictionary with training history
        """
        
        # Create checkpoint directory
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training {self.model.model_name} for {epochs} epochs on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            train_loss = train_metrics['loss']
            self.train_losses.append(train_loss)
            
            # Validation
            if val_loader:
                val_metrics = self.validate_epoch(val_loader)
                val_loss = val_metrics['loss']
                self.val_losses.append(val_loss)
                
                print(f"Epoch {epoch + 1}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
                
                # Early stopping check
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_loss = val_loss
                    
                    # Save best model
                    self.save_checkpoint(
                        checkpoint_dir / f"{self.model.model_name}_best.pth",
                        epoch,
                        val_loss,
                        is_best=True
                    )
                else:
                    patience_counter += 1
                
                # Check early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Log to W&B
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss
                    })
                
                # Update scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
            else:
                print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}")
                
                # Log to W&B
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss
                    })
                
                # Update scheduler
                if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    checkpoint_dir / f"{self.model.model_name}_epoch_{epoch + 1}.pth",
                    epoch,
                    train_loss
                )
            
            # Update model training history
            self.model.update_training_history(
                epoch + 1,
                train_loss,
                val_loss if val_loader else None
            )
        
        # Training completed
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model
        self.save_checkpoint(
            checkpoint_dir / f"{self.model.model_name}_final.pth",
            self.current_epoch,
            self.train_losses[-1] if self.train_losses else 0
        )
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss,
            'total_time': total_time
        }
    
    def save_checkpoint(
        self,
        filepath: Path,
        epoch: int,
        loss: float,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
            'model_info': self.model.get_model_info(),
            'is_best': is_best
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"New best model saved: {filepath}")
            
        # Save to W&B artifacts if enabled
        if self.use_wandb and self.save_artifacts and hasattr(self, 'use_wandb'):
            self._save_wandb_artifact(filepath, is_best, epoch, loss)
    
    def load_checkpoint(self, filepath: Path) -> Dict[str, Any]:
        """Load model checkpoint."""
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Checkpoint loaded from: {filepath}")
        print(f"Epoch: {self.current_epoch}, Loss: {checkpoint.get('loss', 'N/A')}")
        
        return checkpoint
    
    def evaluate_anomaly_detection(
        self,
        test_loader: DataLoader,
        normal_label: int = 0
    ) -> Dict[str, Any]:
        """
        Evaluate anomaly detection performance.
        
        Args:
            test_loader: Test data loader with labels
            normal_label: Label value for normal samples
            
        Returns:
            Evaluation results
        """
        self.model.eval()
        
        all_scores = []
        all_labels = []
        
        print("Computing anomaly scores...")
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluation"):
                images = images.to(self.device)
                
                # Compute anomaly scores
                scores = self.model.compute_anomaly_score(images, reduction='mean')
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Convert labels to binary (normal=0, anomaly=1)
        binary_labels = (all_labels != normal_label).astype(int)
        
        # Evaluate metrics
        results = self.metrics.evaluate_anomaly_detection(binary_labels, all_scores)
        
        print(f"AUC-ROC: {results['auc_roc']:.4f}")
        print(f"AUC-PR: {results['auc_pr']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        
        # Log to W&B
        if self.use_wandb:
            import wandb
            wandb.log(results)
        
        return {
            'scores': all_scores,
            'labels': binary_labels,
            'metrics': results
        }
    
    def _save_wandb_artifact(
        self,
        filepath: Path,
        is_best: bool,
        epoch: int,
        loss: float
    ):
        """Save model checkpoint as W&B artifact."""
        try:
            import wandb
            
            if not wandb.run:
                return
                
            # Create artifact name
            model_name = self.model.model_name
            artifact_name = f"{model_name}-checkpoint"
            
            # Add best suffix for best models
            if is_best:
                artifact_name = f"{model_name}-best-model"
            
            # Create artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"{model_name} checkpoint at epoch {epoch} with loss {loss:.4f}",
                metadata={
                    "epoch": epoch,
                    "loss": loss,
                    "is_best": is_best,
                    "model_type": model_name,
                    "parameters": self.model.count_parameters(),
                    "device": str(self.device),
                    "architecture": self.model.get_model_info()
                }
            )
            
            # Add checkpoint file to artifact
            artifact.add_file(str(filepath))
            
            # Add model summary if available
            try:
                summary_path = filepath.parent / f"{model_name}_summary.txt"
                with open(summary_path, 'w') as f:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Parameters: {self.model.count_parameters():,}\n")
                    f.write(f"Epoch: {epoch}\n")
                    f.write(f"Loss: {loss:.6f}\n")
                    f.write(f"Best model: {is_best}\n")
                    f.write(f"Architecture: {self.model.get_model_info()}\n")
                
                artifact.add_file(str(summary_path))
                
                # Clean up summary file
                if summary_path.exists():
                    summary_path.unlink()
                    
            except Exception as e:
                print(f"Warning: Could not create model summary: {e}")
            
            # Log artifact
            wandb.log_artifact(artifact)
            
            status = "best " if is_best else ""
            print(f"✅ {status}Model artifact saved to W&B: {artifact_name}")
            
        except Exception as e:
            print(f"⚠️  Failed to save W&B artifact: {e}")
    
    def get_learning_curves(self) -> Dict[str, List[float]]:
        """Get training and validation learning curves."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def plot_learning_curves(self, save_path: Optional[str] = None):
        """Plot learning curves."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, label='Training Loss', linewidth=2)
        
        if self.val_losses:
            ax.plot(epochs, self.val_losses, label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{self.model.model_name} Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig