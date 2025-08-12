"""
HuggingFace datasets client for streaming medical images.

Alternative to Kaggle API for dataset access with better streaming capabilities.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Iterator
import pandas as pd
from PIL import Image
import torch
from datasets import load_dataset, DatasetDict, Dataset
import io

from .memory_manager import StreamingMemoryManager


class HuggingFaceStreamClient:
    """
    Stream medical image datasets from HuggingFace Hub.
    
    Supports various medical imaging datasets like:
    - NIH Chest X-ray
    - MIMIC-CXR  
    - ChestX-ray14
    - COVID chest X-ray
    - Custom medical datasets
    """
    
    def __init__(
        self,
        dataset_name: str = "alkzar90/NIH-Chest-X-ray-dataset",
        cache_dir: Optional[str] = None,
        streaming: bool = True,
        token: Optional[str] = None
    ):
        """
        Initialize HuggingFace client.
        
        Args:
            dataset_name: HF dataset identifier
            cache_dir: Local cache directory
            streaming: Enable streaming mode for large datasets
            token: HF authentication token for private datasets
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir or "./hf_cache"
        self.streaming = streaming
        self.token = token
        
        # Dataset splits
        self.dataset = None
        self.splits = []
        
        # Initialize dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from HuggingFace Hub."""
        try:
            print(f"Loading HF dataset: {self.dataset_name}")
            
            self.dataset = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir,
                streaming=self.streaming,
                token=self.token
            )
            
            # Get available splits
            if isinstance(self.dataset, DatasetDict):
                self.splits = list(self.dataset.keys())
            else:
                self.splits = ['train']  # Single split dataset
            
            print(f"✅ Dataset loaded with splits: {self.splits}")
            
        except Exception as e:
            print(f"❌ Failed to load dataset {self.dataset_name}: {e}")
            print("Available alternatives:")
            print("- alkzar90/NIH-Chest-X-ray-dataset")
            print("- keremberke/chest-xray-classification")
            print("- Falah/Alzheimer_MRI")
            raise
    
    def get_available_datasets(self) -> List[str]:
        """Get list of popular medical imaging datasets."""
        return [
            "keremberke/chest-xray-classification",
            "alkzar90/NIH-Chest-X-ray-dataset",
            "Francesco/chest-xray-pneumonia-detection", 
            "Loie/chest-xray-pnemonia-detection",
            "andrewmvd/medical-mnist",
            "manas140/chest-xray",
            "mehdidc/chest_xray",
            "nielsr/chest-xray-pneumonia"
        ]
    
    def get_dataset_info(self) -> Dict:
        """Get dataset information and statistics."""
        if not self.dataset:
            return {}
            
        info = {
            'dataset_name': self.dataset_name,
            'splits': self.splits,
            'streaming': self.streaming
        }
        
        # Get sample counts per split
        for split in self.splits:
            try:
                if self.streaming:
                    # For streaming datasets, we can't get exact count easily
                    info[f'{split}_samples'] = "streaming (unknown count)"
                else:
                    info[f'{split}_samples'] = len(self.dataset[split])
            except:
                info[f'{split}_samples'] = "unknown"
        
        # Get sample data structure
        try:
            sample = next(iter(self.dataset[self.splits[0]]))
            info['columns'] = list(sample.keys())
            info['sample_keys'] = sample.keys()
        except:
            info['columns'] = "unknown"
        
        return info
    
    def stream_images(
        self, 
        split: str = "train",
        max_samples: Optional[int] = None,
        filter_labels: Optional[List[str]] = None
    ) -> Iterator[Dict]:
        """
        Stream images from dataset.
        
        Args:
            split: Dataset split to stream from
            max_samples: Maximum number of samples to stream
            filter_labels: Filter by specific labels/classes
            
        Yields:
            Dict with 'image', 'label', 'metadata' keys
        """
        if split not in self.splits:
            raise ValueError(f"Split '{split}' not available. Available: {self.splits}")
        
        dataset_split = self.dataset[split]
        count = 0
        
        for sample in dataset_split:
            # Check max samples limit
            if max_samples and count >= max_samples:
                break
            
            try:
                # Extract image and label from sample
                image_data = self._extract_image(sample)
                label_data = self._extract_label(sample)
                
                # Filter by labels if specified
                if filter_labels and label_data.get('label_name') not in filter_labels:
                    continue
                
                yield {
                    'image': image_data['image'],
                    'label': label_data,
                    'metadata': {
                        'index': count,
                        'split': split,
                        'dataset': self.dataset_name,
                        **image_data.get('metadata', {}),
                        **{k: v for k, v in sample.items() 
                           if k not in ['image', 'label', 'labels']}
                    }
                }
                
                count += 1
                
            except Exception as e:
                print(f"⚠️ Skipping sample {count}: {e}")
                continue
    
    def _extract_image(self, sample: Dict) -> Dict:
        """Extract image from sample."""
        image_key = self._find_image_key(sample)
        
        if not image_key:
            raise ValueError("No image found in sample")
        
        image_data = sample[image_key]
        
        # Handle different image formats
        if isinstance(image_data, Image.Image):
            image = image_data
        elif isinstance(image_data, dict) and 'bytes' in image_data:
            # Image stored as bytes
            image = Image.open(io.BytesIO(image_data['bytes']))
        elif isinstance(image_data, str):
            # Image path or URL
            image = Image.open(image_data)
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")
        
        # Convert to grayscale if needed (for medical images)
        if image.mode != 'L':
            image = image.convert('L')
        
        return {
            'image': image,
            'metadata': {
                'original_mode': image.mode,
                'size': image.size
            }
        }
    
    def _extract_label(self, sample: Dict) -> Dict:
        """Extract label information from sample."""
        # Try different common label keys
        label_keys = ['label', 'labels', 'class', 'target', 'category']
        
        for key in label_keys:
            if key in sample:
                label = sample[key]
                
                # Handle different label formats
                if isinstance(label, (int, float)):
                    return {
                        'label': int(label),
                        'label_name': str(label)
                    }
                elif isinstance(label, str):
                    return {
                        'label': self._label_to_int(label),
                        'label_name': label
                    }
                elif isinstance(label, dict):
                    # ClassLabel format
                    return {
                        'label': label.get('id', 0),
                        'label_name': label.get('name', 'unknown')
                    }
        
        # Default: assume normal (0)
        return {'label': 0, 'label_name': 'normal'}
    
    def _find_image_key(self, sample: Dict) -> Optional[str]:
        """Find image key in sample."""
        image_keys = ['image', 'img', 'picture', 'photo', 'x_ray', 'xray']
        
        for key in image_keys:
            if key in sample:
                return key
        
        # If no standard key found, try the first key that might be an image
        for key, value in sample.items():
            if isinstance(value, Image.Image):
                return key
            elif isinstance(value, dict) and 'bytes' in value:
                return key
        
        return None
    
    def _label_to_int(self, label: str) -> int:
        """Convert string label to integer."""
        # Common medical image labels
        label_mapping = {
            'normal': 0, 'healthy': 0, 'no finding': 0,
            'pneumonia': 1, 'abnormal': 1, 'disease': 1,
            'covid': 2, 'covid-19': 2,
            'tuberculosis': 3, 'tb': 3
        }
        
        return label_mapping.get(label.lower(), 0)
    
    def filter_normal_images(
        self, 
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> List[Dict]:
        """Filter normal/healthy images."""
        normal_samples = []
        
        for sample in self.stream_images(split, max_samples):
            if sample['label']['label'] == 0:  # Normal
                normal_samples.append(sample)
        
        return normal_samples
    
    def filter_abnormal_images(
        self, 
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> List[Dict]:
        """Filter abnormal/disease images."""
        abnormal_samples = []
        
        for sample in self.stream_images(split, max_samples):
            if sample['label']['label'] != 0:  # Abnormal
                abnormal_samples.append(sample)
        
        return abnormal_samples
    
    def get_class_distribution(self, split: str = "train") -> Dict[str, int]:
        """Get distribution of classes in dataset."""
        distribution = {}
        
        # Sample a subset to estimate distribution
        for i, sample in enumerate(self.stream_images(split, max_samples=1000)):
            label_name = sample['label']['label_name']
            distribution[label_name] = distribution.get(label_name, 0) + 1
            
            if i % 100 == 0:
                print(f"Processed {i} samples for distribution analysis...")
        
        return distribution
    
    def download_sample_dataset(
        self, 
        output_dir: str = "./sample_hf_data",
        max_samples: int = 100
    ):
        """Download a small sample dataset for testing."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        samples_info = []
        
        for i, sample in enumerate(self.stream_images(max_samples=max_samples)):
            if i >= max_samples:
                break
            
            # Save image
            image_filename = f"sample_{i:04d}_{sample['label']['label_name']}.png"
            image_path = output_path / image_filename
            sample['image'].save(image_path)
            
            # Store sample info
            samples_info.append({
                'filename': image_filename,
                'label': sample['label']['label'],
                'label_name': sample['label']['label_name'],
                'size': sample['image'].size,
                **sample['metadata']
            })
            
            if i % 10 == 0:
                print(f"Downloaded {i+1}/{max_samples} samples...")
        
        # Save metadata
        df = pd.DataFrame(samples_info)
        df.to_csv(output_path / "metadata.csv", index=False)
        
        print(f"✅ Sample dataset saved to: {output_path}")
        print(f"Files: {len(samples_info)} images + metadata.csv")
        
        return output_path
    
    def cleanup_cache(self):
        """Clean up local cache."""
        if os.path.exists(self.cache_dir):
            import shutil
            try:
                shutil.rmtree(self.cache_dir)
                print(f"✅ Cleaned cache: {self.cache_dir}")
            except Exception as e:
                print(f"⚠️ Failed to clean cache: {e}")


def test_huggingface_client():
    """Test HuggingFace client functionality."""
    print("🤗 Testing HuggingFace Client")
    print("=" * 40)
    
    try:
        # Initialize client
        client = HuggingFaceStreamClient(
            dataset_name="keremberke/chest-xray-classification",
            streaming=True
        )
        
        # Get dataset info
        info = client.get_dataset_info()
        print("Dataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test streaming
        print("\nTesting streaming (first 5 samples):")
        for i, sample in enumerate(client.stream_images(max_samples=5)):
            print(f"Sample {i}:")
            print(f"  Image size: {sample['image'].size}")
            print(f"  Label: {sample['label']}")
            print(f"  Metadata keys: {list(sample['metadata'].keys())}")
            
            if i >= 4:
                break
        
        # Test class distribution
        print("\nClass distribution (first 100 samples):")
        distribution = client.get_class_distribution()
        for class_name, count in distribution.items():
            print(f"  {class_name}: {count}")
        
        print("\n✅ HuggingFace client test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nTrying alternative dataset...")
        
        try:
            client = HuggingFaceStreamClient(
                dataset_name="Francesco/chest-xray-pneumonia-detection",
                streaming=True
            )
            print("✅ Alternative dataset loaded successfully!")
        except Exception as e2:
            print(f"❌ Alternative also failed: {e2}")


if __name__ == "__main__":
    test_huggingface_client()