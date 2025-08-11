"""
Streaming trainer for memory-efficient training on large datasets.
"""

from typing import Optional, Dict, Any
from pathlib import Path

from .trainer import Trainer
from ..streaming import KaggleStreamClient, StreamingMemoryManager
from ..data import create_streaming_dataloaders


class StreamingTrainer(Trainer):
    """
    Trainer with streaming capabilities for large datasets.
    
    Extends the base Trainer with memory management and progressive
    training across dataset stages.
    """
    
    def __init__(
        self,
        model,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        wandb_project: Optional[str] = None,
        kaggle_client: Optional[KaggleStreamClient] = None,
        memory_manager: Optional[StreamingMemoryManager] = None
    ):
        super().__init__(model, config, device, wandb_project)
        
        # Streaming components
        self.kaggle_client = kaggle_client
        self.memory_manager = memory_manager or StreamingMemoryManager(
            memory_limit_gb=config.get('streaming', {}).get('memory_limit_gb', 4.0) if config else 4.0
        )
        
        # Register cleanup for model cache
        self.memory_manager.register_cleanup_callback(
            "model_cache", 
            self._cleanup_model_cache
        )
    
    def _cleanup_model_cache(self):
        """Cleanup model-specific cache."""
        # Clear any cached computations
        if hasattr(self.model, '_cleanup_cache'):
            self.model._cleanup_cache()
        return True
    
    def train_progressive_stages(
        self,
        stages: list = None,
        epochs_per_stage: int = 5,
        max_samples_per_stage: int = 2000,
        checkpoint_dir: str = "checkpoints",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train progressively across multiple dataset stages.
        
        Args:
            stages: List of stage names to train on
            epochs_per_stage: Epochs per stage
            max_samples_per_stage: Max samples per stage
            checkpoint_dir: Checkpoint directory
            **kwargs: Additional training arguments
            
        Returns:
            Training history across all stages
        """
        if not self.kaggle_client:
            raise ValueError("KaggleStreamClient required for progressive training")
        
        stages = stages or ["images_001", "images_002", "images_003"]
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🌊 Starting progressive streaming training across {len(stages)} stages")
        
        all_history = {
            'stages': [],
            'stage_histories': {},
            'total_time': 0,
            'best_loss': float('inf')
        }
        
        for stage_idx, stage in enumerate(stages):
            print(f"\n📊 Stage {stage_idx + 1}/{len(stages)}: {stage}")
            
            try:
                # Create data loaders for current stage
                train_loader, val_loader, dataset_info = create_streaming_dataloaders(
                    kaggle_client=self.kaggle_client,
                    stage=stage,
                    max_samples=max_samples_per_stage,
                    memory_manager=self.memory_manager,
                    **kwargs
                )
                
                print(f"  📈 Training samples: {dataset_info['train_samples']}")
                print(f"  📉 Validation samples: {dataset_info['val_samples']}")
                
                # Resume from previous stage if available
                resume_path = None
                if stage_idx > 0:
                    prev_checkpoint = checkpoint_dir / f"stage_{stage_idx}_final.pth"
                    if prev_checkpoint.exists():
                        resume_path = prev_checkpoint
                        print(f"  📁 Resuming from: {resume_path}")
                
                if resume_path:
                    self.load_checkpoint(resume_path)
                
                # Train current stage
                stage_history = self.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs_per_stage,
                    checkpoint_dir=str(checkpoint_dir),
                    save_every=max(1, epochs_per_stage // 2),
                    **kwargs
                )
                
                # Save stage-specific checkpoint
                stage_checkpoint = checkpoint_dir / f"stage_{stage_idx + 1}_final.pth"
                self.save_checkpoint(
                    stage_checkpoint,
                    self.current_epoch,
                    stage_history['train_losses'][-1] if stage_history['train_losses'] else 0
                )
                
                # Update overall history
                all_history['stages'].append(stage)
                all_history['stage_histories'][stage] = stage_history
                all_history['total_time'] += stage_history['total_time']
                
                if stage_history['best_loss'] < all_history['best_loss']:
                    all_history['best_loss'] = stage_history['best_loss']
                    
                    # Save overall best model
                    best_checkpoint = checkpoint_dir / f"{self.model.model_name}_best_overall.pth"
                    self.save_checkpoint(
                        best_checkpoint,
                        self.current_epoch,
                        stage_history['best_loss'],
                        is_best=True
                    )
                
                print(f"  ✅ Stage {stage} completed")
                print(f"     Best loss: {stage_history['best_loss']:.4f}")
                print(f"     Time: {stage_history['total_time']:.2f}s")
                
                # Stage cleanup
                self.memory_manager.stage_completed(stage)
                
            except Exception as e:
                print(f"  ❌ Stage {stage} failed: {e}")
                # Continue with next stage
                continue
        
        print(f"\n🎉 Progressive training completed!")
        print(f"   Stages completed: {len(all_history['stages'])}")
        print(f"   Overall best loss: {all_history['best_loss']:.4f}")
        print(f"   Total time: {all_history['total_time']:.2f}s")
        
        return all_history
    
    @classmethod
    def from_config(
        cls, 
        config_path: str,
        model_type: str = "unet",
        **kwargs
    ):
        """
        Create StreamingTrainer from configuration file.
        
        Args:
            config_path: Path to configuration file
            model_type: Type of model to create
            **kwargs: Additional arguments
            
        Returns:
            Configured StreamingTrainer
        """
        from ..utils import ConfigManager
        from .. import create_model
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        # Setup Kaggle client
        try:
            config_manager.setup_kaggle_api()
            kaggle_client = KaggleStreamClient()
        except Exception as e:
            print(f"Warning: Could not setup Kaggle client: {e}")
            kaggle_client = None
        
        # Create model
        model = create_model(model_type, config.model)
        
        # Create trainer
        trainer = cls(
            model=model,
            config=config.__dict__ if hasattr(config, '__dict__') else config,
            device=config_manager.get_device(),
            wandb_project=getattr(config, 'wandb_project', None),
            kaggle_client=kaggle_client,
            **kwargs
        )
        
        # Setup training
        trainer.setup_training(
            learning_rate=config.training.learning_rate,
            optimizer_name=config.training.optimizer,
            scheduler_name=getattr(config.training, 'scheduler', None)
        )
        
        return trainer
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'memory_manager') and self.memory_manager:
            self.memory_manager.stop_monitoring()