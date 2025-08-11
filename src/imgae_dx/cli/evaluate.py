"""
Evaluation command-line interface.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import torch

from ..utils import ConfigManager
from ..models import UNet, ReversedAutoencoder
from ..training import Evaluator
from ..streaming import KaggleStreamClient, StreamingMemoryManager
from ..data import create_test_dataloader


def evaluate_command():
    """Main evaluation command entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate ImgAE-Dx models for anomaly detection performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Paths to multiple model checkpoints for comparison"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["unet", "reversed_ae"],
        help="Model architecture type (auto-detected from checkpoint if not specified)"
    )
    
    # Data arguments
    parser.add_argument(
        "--test-stage",
        type=str,
        default="images_002",
        help="Dataset stage to use for testing"
    )
    
    parser.add_argument(
        "--normal-samples",
        type=int,
        default=500,
        help="Number of normal samples for testing"
    )
    
    parser.add_argument(
        "--abnormal-samples",
        type=int,
        default=500,
        help="Number of abnormal samples for testing"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Input image size"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save evaluation plots"
    )
    
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save detailed evaluation report"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for evaluation"
    )
    
    # Memory management
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=4.0,
        help="Memory limit in GB"
    )
    
    # Analysis options
    parser.add_argument(
        "--analyze-latent",
        action="store_true",
        help="Perform latent space analysis"
    )
    
    parser.add_argument(
        "--visualize-reconstructions",
        action="store_true",
        help="Generate reconstruction visualizations"
    )
    
    parser.add_argument(
        "--error-maps",
        action="store_true",
        help="Generate error maps"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model and not args.models:
        parser.error("Either --model or --models must be specified")
    
    # Execute evaluation
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main(args):
    """Main evaluation logic."""
    print("📊 ImgAE-Dx Model Evaluation")
    print("=" * 40)
    
    # Setup configuration
    config_manager = ConfigManager()
    
    # Setup device
    if args.device == "auto":
        device = config_manager.get_device()
    else:
        device = args.device
    
    print(f"🖥️  Using device: {device}")
    
    # Setup memory management
    memory_manager = StreamingMemoryManager(
        memory_limit_gb=args.memory_limit,
        enable_monitoring=True
    )
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Results will be saved to: {output_dir}")
    
    # Setup Kaggle client
    try:
        kaggle_client = KaggleStreamClient()
        print("✅ Kaggle client initialized")
    except Exception as e:
        print(f"⚠️  Kaggle client error: {e}")
        print("Using dummy data for testing...")
        kaggle_client = None
    
    # Create test data loader
    if kaggle_client:
        test_loader, dataset_info = create_test_dataloader(
            kaggle_client=kaggle_client,
            stage=args.test_stage,
            batch_size=args.batch_size,
            normal_samples=args.normal_samples,
            abnormal_samples=args.abnormal_samples,
            memory_manager=memory_manager,
            image_size=args.image_size
        )
    else:
        # Create dummy test data
        test_loader = create_dummy_test_dataloader(
            batch_size=args.batch_size,
            image_size=args.image_size,
            normal_samples=args.normal_samples,
            abnormal_samples=args.abnormal_samples
        )
        dataset_info = {
            'normal_samples': args.normal_samples,
            'abnormal_samples': args.abnormal_samples,
            'total_samples': args.normal_samples + args.abnormal_samples
        }
    
    print(f"✅ Test data loaded:")
    print(f"   Normal:   {dataset_info['normal_samples']} samples")
    print(f"   Abnormal: {dataset_info['abnormal_samples']} samples")
    print(f"   Total:    {dataset_info['total_samples']} samples")
    
    # Setup evaluator
    evaluator = Evaluator(device=device)
    
    # Load models
    models = {}
    
    if args.model:
        # Single model evaluation
        model_name, model = load_model_from_checkpoint(
            args.model, args.model_type, device
        )
        models[model_name] = model
        print(f"✅ Loaded model: {model_name}")
        
    elif args.models:
        # Multiple models comparison
        for model_path in args.models:
            model_name, model = load_model_from_checkpoint(
                model_path, args.model_type, device
            )
            models[model_name] = model
            print(f"✅ Loaded model: {model_name}")
    
    print(f"\n🔍 Evaluating {len(models)} model(s)...")
    
    # Perform evaluation
    if len(models) == 1:
        # Single model evaluation
        model_name = list(models.keys())[0]
        model = models[model_name]
        
        results = evaluator.evaluate_single_model(
            model=model,
            test_loader=test_loader,
            normal_label=0,
            model_name=model_name
        )
        
        print_evaluation_results(results)
        
        # Save results
        if args.save_plots or args.save_report:
            save_single_model_results(
                results, output_dir, args, evaluator
            )
        
    else:
        # Multiple models comparison
        comparison_results = evaluator.compare_models(
            models=models,
            test_loader=test_loader,
            normal_label=0,
            save_dir=str(output_dir) if args.save_plots else None
        )
        
        print_comparison_results(comparison_results)
        
        # Additional analysis
        if args.analyze_latent:
            print("\n🧠 Analyzing latent space...")
            latent_analysis = evaluator.analyze_latent_space(
                comparison_results['individual_results'],
                save_path=str(output_dir / "latent_analysis.png") if args.save_plots else None
            )
        
        if args.visualize_reconstructions:
            print("\n🖼️  Generating reconstruction visualizations...")
            evaluator.visualize_reconstructions(
                comparison_results['individual_results'],
                save_path=str(output_dir / "reconstructions.png") if args.save_plots else None
            )
        
        if args.error_maps:
            print("\n🗺️  Generating error maps...")
            evaluator.generate_error_maps(
                comparison_results['individual_results'],
                save_path=str(output_dir / "error_maps.png") if args.save_plots else None
            )
        
        # Save comprehensive report
        if args.save_report:
            report = evaluator.generate_comprehensive_report(
                comparison_results,
                save_path=str(output_dir / "evaluation_report.md")
            )
            print(f"\n📋 Detailed report saved to: {output_dir / 'evaluation_report.md'}")
    
    print(f"\n🎉 Evaluation completed! Results saved to: {output_dir}")
    
    # Cleanup
    memory_manager.stop_monitoring()
    if kaggle_client:
        kaggle_client.cleanup_cache()


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_type: str = None,
    device: str = "cpu"
) -> tuple:
    """Load model from checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine model type from checkpoint or filename
    if model_type is None:
        if 'model_info' in checkpoint and 'architecture' in checkpoint['model_info']:
            arch = checkpoint['model_info']['architecture'].lower()
            if 'unet' in arch:
                model_type = "unet"
            elif 'reversed' in arch or 'ra' in arch:
                model_type = "reversed_ae"
        else:
            # Try to infer from filename
            filename = checkpoint_path.stem.lower()
            if 'unet' in filename:
                model_type = "unet"
            elif 'reversed' in filename or 'ra' in filename:
                model_type = "reversed_ae"
            else:
                raise ValueError("Cannot determine model type. Please specify --model-type")
    
    # Get model configuration from checkpoint
    model_config = checkpoint.get('model_info', {})
    
    # Create model
    if model_type == "unet":
        model = UNet(
            input_channels=model_config.get('input_channels', 1),
            input_size=model_config.get('input_size', 128),
            latent_dim=model_config.get('latent_dim', 512)
        )
    elif model_type == "reversed_ae":
        model = ReversedAutoencoder(
            input_channels=model_config.get('input_channels', 1),
            input_size=model_config.get('input_size', 128),
            latent_dim=model_config.get('latent_dim', 512)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Generate model name
    model_name = f"{model_type}_{checkpoint_path.stem}"
    
    return model_name, model


def create_dummy_test_dataloader(
    batch_size: int,
    image_size: int,
    normal_samples: int,
    abnormal_samples: int
):
    """Create dummy test data loader."""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create normal samples (label 0)
    normal_images = torch.randn(normal_samples, 1, image_size, image_size)
    normal_labels = torch.zeros(normal_samples, dtype=torch.long)
    
    # Create abnormal samples (label 1) with higher variance
    abnormal_images = torch.randn(abnormal_samples, 1, image_size, image_size) * 1.5
    abnormal_labels = torch.ones(abnormal_samples, dtype=torch.long)
    
    # Combine datasets
    all_images = torch.cat([normal_images, abnormal_images], dim=0)
    all_labels = torch.cat([normal_labels, abnormal_labels], dim=0)
    
    test_dataset = TensorDataset(all_images, all_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("⚠️  Using dummy test data")
    
    return test_loader


def print_evaluation_results(results: Dict[str, Any]):
    """Print single model evaluation results."""
    metrics = results['metrics']
    stats = results['score_stats']
    
    print(f"\n📊 Evaluation Results - {results['model_name']}")
    print("=" * 50)
    print(f"AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"AUC-PR:       {metrics['auc_pr']:.4f}")
    print(f"F1-Score:     {metrics['f1_score']:.4f}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Sensitivity:  {metrics['sensitivity']:.4f}")
    print(f"Specificity:  {metrics['specificity']:.4f}")
    print(f"Threshold:    {metrics['threshold']:.4f}")
    print()
    print("Score Statistics:")
    print(f"  Normal mean:   {stats['normal_mean']:.4f} ± {stats['normal_std']:.4f}")
    print(f"  Anomaly mean:  {stats['anomaly_mean']:.4f} ± {stats['anomaly_std']:.4f}")
    print(f"  Separation:    {stats['separation']:.4f}")


def print_comparison_results(comparison_results: Dict[str, Any]):
    """Print model comparison results."""
    print(f"\n📊 Model Comparison Results")
    print("=" * 50)
    print(f"Best model: {comparison_results['best_model']['name']} "
          f"(AUC-ROC: {comparison_results['best_model']['auc_roc']:.4f})")
    print()
    print("Model Rankings (by AUC-ROC):")
    print("-" * 30)
    
    for i, (name, metrics) in enumerate(comparison_results['ranking'], 1):
        print(f"{i:2d}. {name:20s} AUC: {metrics['auc_roc']:.4f} "
              f"F1: {metrics['f1_score']:.4f}")


def save_single_model_results(
    results: Dict[str, Any],
    output_dir: Path,
    args,
    evaluator: Evaluator
):
    """Save results for single model evaluation."""
    model_name = results['model_name']
    
    if args.save_plots:
        # ROC curve
        evaluator.metrics.plot_roc_curve(
            results['labels'],
            results['scores'],
            save_path=str(output_dir / f"{model_name}_roc_curve.png")
        )
        
        # PR curve
        evaluator.metrics.plot_precision_recall_curve(
            results['labels'],
            results['scores'],
            save_path=str(output_dir / f"{model_name}_pr_curve.png")
        )
        
        # Score distribution
        evaluator.metrics.plot_score_distribution(
            results['normal_scores'],
            results['anomaly_scores'],
            save_path=str(output_dir / f"{model_name}_score_dist.png")
        )
        
        # Confusion matrix
        y_pred = (results['scores'] >= results['metrics']['threshold']).astype(int)
        evaluator.metrics.plot_confusion_matrix(
            results['labels'],
            y_pred,
            save_path=str(output_dir / f"{model_name}_confusion_matrix.png")
        )
    
    if args.save_report:
        # Save detailed report
        report = evaluator.metrics.get_summary_report(results['metrics'])
        with open(output_dir / f"{model_name}_report.txt", 'w') as f:
            f.write(report)
        
        # Save metrics as JSON
        evaluator.metrics.save_results(
            results['metrics'],
            str(output_dir / f"{model_name}_metrics.json")
        )


if __name__ == "__main__":
    evaluate_command()