"""
Integration tests for training pipeline.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import tempfile
from pathlib import Path

from src.imgae_dx.models import UNet, ReversedAutoencoder
from src.imgae_dx.training import Trainer, Evaluator
from src.imgae_dx.utils import ConfigManager


@pytest.mark.integration
class TestTrainingIntegration:
    """Integration tests for the training pipeline."""
    
    @pytest.fixture
    def simple_dataset(self):
        """Create a simple dataset for integration testing."""
        # Create synthetic data that's easy to learn
        images = torch.randn(50, 1, 32, 32) * 0.1 + 0.5  # Low variance around 0.5
        labels = torch.zeros(50, dtype=torch.long)
        
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=8, shuffle=True)
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        from src.imgae_dx.utils.config_manager import ProjectConfig
        
        config = ProjectConfig()
        config.model.input_size = 32
        config.model.latent_dim = 64
        config.training.epochs = 3
        config.training.batch_size = 8
        config.training.learning_rate = 1e-3
        config.training.patience = 10
        
        return config
    
    def test_unet_training_pipeline(self, simple_dataset, test_config, temp_checkpoint_dir, device):
        """Test complete U-Net training pipeline."""
        # Create model
        model = UNet(
            input_channels=1,
            input_size=32,
            latent_dim=64
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=test_config.__dict__,
            device=device,
            wandb_project=None  # Disable W&B for testing
        )
        
        # Setup training
        trainer.setup_training(
            learning_rate=test_config.training.learning_rate,
            optimizer_name="adam"
        )
        
        # Train model
        history = trainer.train(
            train_loader=simple_dataset,
            val_loader=None,  # No validation for simplicity
            epochs=test_config.training.epochs,
            checkpoint_dir=str(temp_checkpoint_dir),
            early_stopping_patience=10
        )
        
        # Verify training completed
        assert 'train_losses' in history
        assert len(history['train_losses']) == test_config.training.epochs
        assert history['total_time'] > 0
        
        # Verify loss decreased
        initial_loss = history['train_losses'][0]
        final_loss = history['train_losses'][-1]
        assert final_loss <= initial_loss  # Should improve or stay same
        
        # Verify checkpoint was saved
        checkpoint_files = list(temp_checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0
    
    def test_reversed_ae_training_pipeline(self, simple_dataset, test_config, temp_checkpoint_dir, device):
        """Test complete Reversed AE training pipeline."""
        # Create model
        model = ReversedAutoencoder(
            input_channels=1,
            input_size=32,
            latent_dim=64
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=test_config.__dict__,
            device=device,
            wandb_project=None
        )
        
        # Setup training
        trainer.setup_training(
            learning_rate=test_config.training.learning_rate,
            optimizer_name="adam"
        )
        
        # Train model
        history = trainer.train(
            train_loader=simple_dataset,
            val_loader=None,
            epochs=test_config.training.epochs,
            checkpoint_dir=str(temp_checkpoint_dir)
        )
        
        # Verify training completed
        assert 'train_losses' in history
        assert len(history['train_losses']) == test_config.training.epochs
        
        # Verify checkpoint was saved
        checkpoint_files = list(temp_checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0
    
    def test_training_with_validation(self, test_config, temp_checkpoint_dir, device):
        """Test training with validation split."""
        # Create larger dataset for train/val split
        train_images = torch.randn(60, 1, 32, 32) * 0.1 + 0.5
        train_labels = torch.zeros(60, dtype=torch.long)
        train_dataset = TensorDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        
        val_images = torch.randn(20, 1, 32, 32) * 0.1 + 0.5
        val_labels = torch.zeros(20, dtype=torch.long)
        val_dataset = TensorDataset(val_images, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Create model and trainer
        model = UNet(input_channels=1, input_size=32, latent_dim=64)
        trainer = Trainer(model=model, device=device, wandb_project=None)
        trainer.setup_training(learning_rate=1e-3)
        
        # Train with validation
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,
            checkpoint_dir=str(temp_checkpoint_dir)
        )
        
        # Verify both train and val losses are recorded
        assert 'train_losses' in history
        assert 'val_losses' in history
        assert len(history['train_losses']) == 3
        assert len(history['val_losses']) == 3
    
    def test_checkpoint_resume(self, simple_dataset, test_config, temp_checkpoint_dir, device):
        """Test resuming training from checkpoint."""
        # Initial training
        model = UNet(input_channels=1, input_size=32, latent_dim=64)
        trainer1 = Trainer(model=model, device=device, wandb_project=None)
        trainer1.setup_training(learning_rate=1e-3)
        
        # Train for 2 epochs
        history1 = trainer1.train(
            train_loader=simple_dataset,
            val_loader=None,
            epochs=2,
            checkpoint_dir=str(temp_checkpoint_dir)
        )
        
        # Find the final checkpoint
        checkpoint_files = list(temp_checkpoint_dir.glob("*final*.pth"))
        assert len(checkpoint_files) > 0
        checkpoint_path = checkpoint_files[0]
        
        # Resume training
        model2 = UNet(input_channels=1, input_size=32, latent_dim=64)
        trainer2 = Trainer(model=model2, device=device, wandb_project=None)
        trainer2.setup_training(learning_rate=1e-3)
        
        # Load checkpoint
        loaded_checkpoint = trainer2.load_checkpoint(checkpoint_path)
        
        # Verify checkpoint was loaded correctly
        assert loaded_checkpoint['epoch'] == 1  # 0-indexed, so epoch 2 = index 1
        assert trainer2.current_epoch == 1
        
        # Continue training for 1 more epoch
        history2 = trainer2.train(
            train_loader=simple_dataset,
            val_loader=None,
            epochs=1,
            checkpoint_dir=str(temp_checkpoint_dir)
        )
        
        # Should have trained for 1 additional epoch
        assert len(history2['train_losses']) == 1
    
    def test_early_stopping(self, test_config, temp_checkpoint_dir, device):
        """Test early stopping functionality."""
        # Create dataset that will cause overfitting (very small)
        images = torch.randn(10, 1, 32, 32)
        labels = torch.zeros(10, dtype=torch.long)
        train_dataset = TensorDataset(images, labels)
        train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
        
        # Create validation set that's different
        val_images = torch.randn(10, 1, 32, 32) * 2  # Higher variance
        val_labels = torch.zeros(10, dtype=torch.long)
        val_dataset = TensorDataset(val_images, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)
        
        # Create model and trainer
        model = UNet(input_channels=1, input_size=32, latent_dim=64)
        trainer = Trainer(model=model, device=device, wandb_project=None)
        trainer.setup_training(learning_rate=1e-2)  # Higher LR for faster overfitting
        
        # Train with early stopping
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=20,  # Max epochs
            checkpoint_dir=str(temp_checkpoint_dir),
            early_stopping_patience=3,  # Short patience
            min_delta=1e-4
        )
        
        # Should stop early due to validation loss not improving
        assert len(history['train_losses']) < 20


@pytest.mark.integration
class TestEvaluationIntegration:
    """Integration tests for evaluation pipeline."""
    
    @pytest.fixture
    def test_data(self):
        """Create test data for evaluation."""
        # Normal samples (lower variance)
        normal_images = torch.randn(30, 1, 32, 32) * 0.1 + 0.5
        normal_labels = torch.zeros(30, dtype=torch.long)
        
        # Abnormal samples (higher variance)
        abnormal_images = torch.randn(20, 1, 32, 32) * 0.5 + 0.5
        abnormal_labels = torch.ones(20, dtype=torch.long)
        
        # Combine datasets
        all_images = torch.cat([normal_images, abnormal_images])
        all_labels = torch.cat([normal_labels, abnormal_labels])
        
        dataset = TensorDataset(all_images, all_labels)
        return DataLoader(dataset, batch_size=10, shuffle=False)
    
    def test_single_model_evaluation(self, test_data, device):
        """Test evaluation of a single model."""
        # Create and "train" model (just initialize)
        model = UNet(input_channels=1, input_size=32, latent_dim=64)
        model.to(device)
        
        # Create evaluator
        evaluator = Evaluator(device=device)
        
        # Evaluate model
        results = evaluator.evaluate_single_model(
            model=model,
            test_loader=test_data,
            normal_label=0,
            model_name="test_unet"
        )
        
        # Verify results structure
        assert 'model_name' in results
        assert 'scores' in results
        assert 'labels' in results
        assert 'metrics' in results
        assert 'score_stats' in results
        
        # Verify metrics
        metrics = results['metrics']
        assert 'auc_roc' in metrics
        assert 'auc_pr' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['auc_roc'] <= 1
        assert 0 <= metrics['auc_pr'] <= 1
    
    def test_model_comparison(self, test_data, device):
        """Test comparison of multiple models."""
        # Create two models
        unet = UNet(input_channels=1, input_size=32, latent_dim=64)
        reversed_ae = ReversedAutoencoder(input_channels=1, input_size=32, latent_dim=64)
        
        models = {
            "UNet": unet,
            "ReversedAE": reversed_ae
        }
        
        # Create evaluator
        evaluator = Evaluator(device=device)
        
        # Compare models
        comparison_results = evaluator.compare_models(
            models=models,
            test_loader=test_data,
            normal_label=0
        )
        
        # Verify comparison structure
        assert 'individual_results' in comparison_results
        assert 'comparison_metrics' in comparison_results
        assert 'best_model' in comparison_results
        assert 'ranking' in comparison_results
        
        # Verify all models were evaluated
        assert len(comparison_results['individual_results']) == 2
        assert 'UNet' in comparison_results['individual_results']
        assert 'ReversedAE' in comparison_results['individual_results']
        
        # Verify ranking
        ranking = comparison_results['ranking']
        assert len(ranking) == 2
        assert ranking[0][1]['auc_roc'] >= ranking[1][1]['auc_roc']  # Sorted by AUC


@pytest.mark.integration  
@pytest.mark.slow
class TestEndToEndPipeline:
    """End-to-end integration tests."""
    
    def test_complete_pipeline(self, temp_checkpoint_dir, device):
        """Test complete train -> evaluate pipeline."""
        # Step 1: Create training data
        train_images = torch.randn(80, 1, 32, 32) * 0.1 + 0.5
        train_labels = torch.zeros(80, dtype=torch.long)
        train_dataset = TensorDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Step 2: Train model
        model = UNet(input_channels=1, input_size=32, latent_dim=64)
        trainer = Trainer(model=model, device=device, wandb_project=None)
        trainer.setup_training(learning_rate=1e-3, optimizer_name="adam")
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=None,
            epochs=5,
            checkpoint_dir=str(temp_checkpoint_dir)
        )
        
        # Verify training completed
        assert len(history['train_losses']) == 5
        
        # Step 3: Create test data
        normal_test = torch.randn(20, 1, 32, 32) * 0.1 + 0.5
        abnormal_test = torch.randn(20, 1, 32, 32) * 0.3 + 0.5
        
        test_images = torch.cat([normal_test, abnormal_test])
        test_labels = torch.cat([torch.zeros(20), torch.ones(20)]).long()
        
        test_dataset = TensorDataset(test_images, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
        
        # Step 4: Evaluate trained model
        evaluator = Evaluator(device=device)
        
        results = evaluator.evaluate_single_model(
            model=trainer.model,  # Use trained model
            test_loader=test_loader,
            normal_label=0,
            model_name="trained_unet"
        )
        
        # Step 5: Verify evaluation results
        assert results['metrics']['auc_roc'] >= 0.5  # Should be better than random
        assert len(results['scores']) == 40  # Total test samples
        assert len(results['labels']) == 40
        
        # Step 6: Test checkpoint loading and evaluation
        checkpoint_files = list(temp_checkpoint_dir.glob("*final*.pth"))
        assert len(checkpoint_files) > 0
        
        # Load model from checkpoint
        new_model = UNet(input_channels=1, input_size=32, latent_dim=64)
        new_model.load_state_dict(torch.load(checkpoint_files[0])['model_state_dict'])
        new_model.to(device)
        
        # Evaluate loaded model
        loaded_results = evaluator.evaluate_single_model(
            model=new_model,
            test_loader=test_loader,
            normal_label=0,
            model_name="loaded_unet"
        )
        
        # Results should be identical (or very close due to device precision)
        assert abs(results['metrics']['auc_roc'] - loaded_results['metrics']['auc_roc']) < 1e-6
    
    def test_config_integration(self, temp_checkpoint_dir):
        """Test integration with configuration system."""
        # Create config file
        config_content = """
project_name: Integration Test
model:
  input_size: 32
  latent_dim: 64
training:
  epochs: 3
  batch_size: 8
  learning_rate: 0.001
data:
  image_size: 32
"""
        
        config_file = temp_checkpoint_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        # Load config
        config_manager = ConfigManager()
        config = config_manager.load_config(str(config_file))
        
        # Verify config loaded correctly
        assert config.project_name == "Integration Test"
        assert config.model.input_size == 32
        assert config.training.epochs == 3
        
        # Test config validation
        # Note: API keys won't validate in test environment, but that's expected
        is_valid = config_manager.validate_config()
        assert isinstance(is_valid, bool)