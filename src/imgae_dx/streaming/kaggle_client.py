"""
Kaggle API client for streaming NIH Chest X-ray dataset.
"""

import os
import io
import zipfile
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any
import pandas as pd
from PIL import Image

try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False


class KaggleStreamClient:
    """
    Streaming client for Kaggle datasets.
    
    Provides memory-efficient access to large datasets like NIH Chest X-ray
    without requiring full local storage.
    """
    
    def __init__(
        self,
        dataset_name: str = "nih-chest-xrays/data",
        temp_dir: Optional[str] = None,
        chunk_size: int = 8192
    ):
        if not KAGGLE_AVAILABLE:
            raise ImportError(
                "Kaggle API not available. Install with: pip install kaggle"
            )
        
        self.dataset_name = dataset_name
        self.chunk_size = chunk_size
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        # Initialize API
        self._setup_api()
        
        # Cache for metadata
        self._metadata_cache: Dict[str, pd.DataFrame] = {}
        self._file_list_cache: Optional[List[str]] = None
    
    def _setup_api(self):
        """Setup and authenticate Kaggle API."""
        try:
            kaggle.api.authenticate()
            self.api = kaggle.api
        except Exception as e:
            raise RuntimeError(f"Failed to authenticate Kaggle API: {e}")
    
    def list_dataset_files(self) -> List[str]:
        """List all files in the dataset."""
        if self._file_list_cache is None:
            try:
                files = self.api.dataset_list_files(self.dataset_name)
                self._file_list_cache = [f.name for f in files.files]
            except Exception as e:
                raise RuntimeError(f"Failed to list dataset files: {e}")
        
        return self._file_list_cache
    
    def get_image_stages(self) -> List[str]:
        """Get available image stage files (ZIP archives)."""
        files = self.list_dataset_files()
        image_files = [
            f for f in files 
            if f.startswith('images_') and f.endswith('.zip')
        ]
        return sorted(image_files)
    
    def stream_zip_content(self, filename: str) -> zipfile.ZipFile:
        """
        Stream ZIP file content directly to memory.
        
        Args:
            filename: Name of the ZIP file to stream
            
        Returns:
            ZipFile object for streaming access
        """
        try:
            # Download to memory buffer
            temp_buffer = io.BytesIO()
            
            # Download file in chunks to memory
            self.api.dataset_download_file(
                self.dataset_name, 
                filename,
                path=self.temp_dir,
                unzip=False
            )
            
            # Read the downloaded file
            temp_filepath = Path(self.temp_dir) / filename
            with open(temp_filepath, 'rb') as f:
                temp_buffer.write(f.read())
            
            # Clean up temporary file
            temp_filepath.unlink()
            
            # Reset buffer position and create ZipFile
            temp_buffer.seek(0)
            return zipfile.ZipFile(temp_buffer)
            
        except Exception as e:
            raise RuntimeError(f"Failed to stream ZIP content for {filename}: {e}")
    
    def get_metadata_dataframe(self, stage: str = "Data_Entry_2017.csv") -> pd.DataFrame:
        """
        Load NIH dataset metadata.
        
        Args:
            stage: Metadata file name
            
        Returns:
            DataFrame with image metadata
        """
        if stage not in self._metadata_cache:
            try:
                # Download metadata file to temporary location
                temp_path = Path(self.temp_dir) / stage
                
                self.api.dataset_download_file(
                    self.dataset_name,
                    stage,
                    path=self.temp_dir,
                    unzip=False
                )
                
                # Load DataFrame
                df = pd.read_csv(temp_path)
                self._metadata_cache[stage] = df
                
                # Clean up
                temp_path.unlink()
                
            except Exception as e:
                raise RuntimeError(f"Failed to load metadata {stage}: {e}")
        
        return self._metadata_cache[stage]
    
    def filter_normal_images(
        self, 
        metadata_df: pd.DataFrame,
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Filter for normal (no findings) images.
        
        Args:
            metadata_df: Full metadata DataFrame
            max_samples: Maximum number of samples to return
            
        Returns:
            Filtered DataFrame with normal images
        """
        # Filter for "No Finding" cases
        normal_df = metadata_df[
            metadata_df['Finding Labels'] == 'No Finding'
        ].copy()
        
        if max_samples is not None:
            normal_df = normal_df.sample(n=min(max_samples, len(normal_df)), 
                                       random_state=42)
        
        return normal_df
    
    def filter_abnormal_images(
        self,
        metadata_df: pd.DataFrame,
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Filter for abnormal (with findings) images.
        
        Args:
            metadata_df: Full metadata DataFrame  
            max_samples: Maximum number of samples to return
            
        Returns:
            Filtered DataFrame with abnormal images
        """
        # Filter for cases with findings (not "No Finding")
        abnormal_df = metadata_df[
            metadata_df['Finding Labels'] != 'No Finding'
        ].copy()
        
        if max_samples is not None:
            abnormal_df = abnormal_df.sample(n=min(max_samples, len(abnormal_df)),
                                           random_state=42)
        
        return abnormal_df
    
    def get_progressive_streams(self) -> Iterator[zipfile.ZipFile]:
        """
        Get progressive ZIP file streams for staged training.
        
        Yields:
            ZipFile objects for each stage
        """
        image_stages = self.get_image_stages()
        
        for stage in image_stages:
            print(f"Streaming stage: {stage}")
            yield self.stream_zip_content(stage)
    
    def extract_image_from_zip(
        self, 
        zip_file: zipfile.ZipFile, 
        image_filename: str
    ) -> Image.Image:
        """
        Extract specific image from ZIP stream.
        
        Args:
            zip_file: ZipFile object
            image_filename: Name of image file within ZIP
            
        Returns:
            PIL Image object
        """
        try:
            with zip_file.open(image_filename) as img_file:
                image = Image.open(img_file).convert('L')  # Convert to grayscale
                return image
                
        except KeyError:
            raise FileNotFoundError(f"Image {image_filename} not found in ZIP")
        except Exception as e:
            raise RuntimeError(f"Failed to extract {image_filename}: {e}")
    
    def get_stage_image_list(
        self, 
        zip_file: zipfile.ZipFile,
        metadata_df: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """
        Get list of image files in a ZIP stage.
        
        Args:
            zip_file: ZipFile object
            metadata_df: Optional metadata to filter images
            
        Returns:
            List of image filenames
        """
        # Get all image files in ZIP
        image_files = [
            name for name in zip_file.namelist()
            if name.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        # Filter based on metadata if provided
        if metadata_df is not None:
            valid_filenames = set(metadata_df['Image Index'].values)
            image_files = [
                f for f in image_files
                if Path(f).name in valid_filenames
            ]
        
        return image_files
    
    def estimate_memory_usage(self, stage: str) -> Dict[str, Any]:
        """
        Estimate memory usage for a given stage.
        
        Args:
            stage: ZIP file name
            
        Returns:
            Dictionary with memory estimates
        """
        try:
            files = self.list_dataset_files()
            file_info = next((f for f in files if f == stage), None)
            
            if file_info is None:
                return {"error": f"Stage {stage} not found"}
            
            # Rough estimates (actual usage may vary)
            zip_size_mb = file_info.get('size', 0) / (1024 * 1024)
            estimated_uncompressed_mb = zip_size_mb * 3  # Rough compression ratio
            
            return {
                "stage": stage,
                "zip_size_mb": zip_size_mb,
                "estimated_uncompressed_mb": estimated_uncompressed_mb,
                "recommended_batch_size": max(1, int(500 / estimated_uncompressed_mb * 32))
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def cleanup_cache(self):
        """Clear cached data to free memory."""
        self._metadata_cache.clear()
        self._file_list_cache = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup_cache()