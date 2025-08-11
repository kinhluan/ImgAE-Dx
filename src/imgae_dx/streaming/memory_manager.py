"""
Memory management for streaming data processing.
"""

import gc
import psutil
import threading
import time
from typing import Dict, Any, Optional, Callable
from collections import deque


class StreamingMemoryManager:
    """
    Memory management system for streaming data processing.
    
    Monitors memory usage and triggers cleanup when thresholds are exceeded.
    """
    
    def __init__(
        self,
        memory_limit_gb: float = 4.0,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.9,
        cleanup_frequency: int = 100,
        enable_monitoring: bool = True
    ):
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.cleanup_frequency = cleanup_frequency
        self.enable_monitoring = enable_monitoring
        
        # Monitoring state
        self._batch_counter = 0
        self._memory_history = deque(maxlen=100)
        self._cleanup_callbacks: Dict[str, Callable] = {}
        
        # Threading for background monitoring
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        if enable_monitoring:
            self.start_monitoring()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            'system_total_gb': system_memory.total / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_used_percent': system_memory.percent,
            'process_rss_gb': process_memory.rss / (1024**3),
            'process_vms_gb': process_memory.vms / (1024**3),
            'memory_limit_gb': self.memory_limit_bytes / (1024**3),
            'limit_usage_percent': (process_memory.rss / self.memory_limit_bytes) * 100
        }
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage and return status."""
        usage = self.get_memory_usage()
        
        # Determine status
        limit_usage = usage['limit_usage_percent']
        
        if limit_usage >= self.critical_threshold * 100:
            status = 'critical'
            action_needed = True
        elif limit_usage >= self.warning_threshold * 100:
            status = 'warning'
            action_needed = True
        else:
            status = 'normal'
            action_needed = False
        
        result = {
            **usage,
            'status': status,
            'action_needed': action_needed,
            'threshold_warning': self.warning_threshold * 100,
            'threshold_critical': self.critical_threshold * 100
        }
        
        # Store in history
        self._memory_history.append(result)
        
        return result
    
    def trigger_cleanup(self, force: bool = False) -> Dict[str, Any]:
        """
        Trigger memory cleanup.
        
        Args:
            force: Force cleanup regardless of memory usage
            
        Returns:
            Dictionary with cleanup results
        """
        before_usage = self.get_memory_usage()
        
        # Call registered cleanup callbacks
        cleanup_results = {}
        for name, callback in self._cleanup_callbacks.items():
            try:
                start_time = time.time()
                result = callback()
                cleanup_results[name] = {
                    'success': True,
                    'duration': time.time() - start_time,
                    'result': result
                }
            except Exception as e:
                cleanup_results[name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Force garbage collection
        gc_collected = gc.collect()
        
        after_usage = self.get_memory_usage()
        
        return {
            'before_rss_gb': before_usage['process_rss_gb'],
            'after_rss_gb': after_usage['process_rss_gb'],
            'memory_freed_gb': before_usage['process_rss_gb'] - after_usage['process_rss_gb'],
            'gc_collected': gc_collected,
            'cleanup_callbacks': cleanup_results,
            'forced': force
        }
    
    def register_cleanup_callback(self, name: str, callback: Callable) -> None:
        """
        Register a cleanup callback function.
        
        Args:
            name: Unique name for the callback
            callback: Function to call during cleanup
        """
        self._cleanup_callbacks[name] = callback
    
    def unregister_cleanup_callback(self, name: str) -> None:
        """Remove a cleanup callback."""
        self._cleanup_callbacks.pop(name, None)
    
    def batch_processed(self) -> Optional[Dict[str, Any]]:
        """
        Call this after processing each batch.
        
        Returns:
            Cleanup results if cleanup was triggered, None otherwise
        """
        self._batch_counter += 1
        
        # Check if cleanup is needed
        if self._batch_counter % self.cleanup_frequency == 0:
            usage = self.check_memory_usage()
            
            if usage['action_needed']:
                return self.trigger_cleanup()
        
        return None
    
    def stage_completed(self, stage_name: str) -> Dict[str, Any]:
        """
        Call this when a data stage is completed.
        
        Args:
            stage_name: Name of the completed stage
            
        Returns:
            Cleanup results
        """
        print(f"Stage '{stage_name}' completed. Performing cleanup...")
        return self.trigger_cleanup(force=True)
    
    def start_monitoring(self) -> None:
        """Start background memory monitoring thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        if self._monitor_thread is not None:
            self._stop_monitoring.set()
            self._monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                usage = self.check_memory_usage()
                
                # Emergency cleanup for critical memory usage
                if usage['status'] == 'critical':
                    print(f"CRITICAL: Memory usage at {usage['limit_usage_percent']:.1f}%")
                    self.trigger_cleanup(force=True)
                
                elif usage['status'] == 'warning':
                    print(f"WARNING: Memory usage at {usage['limit_usage_percent']:.1f}%")
                
                # Wait before next check
                self._stop_monitoring.wait(30.0)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Error in memory monitoring: {e}")
                self._stop_monitoring.wait(60.0)  # Wait longer after error
    
    def get_memory_history(self) -> list:
        """Get recent memory usage history."""
        return list(self._memory_history)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self._memory_history:
            return {'error': 'No memory history available'}
        
        recent_usage = [entry['limit_usage_percent'] for entry in self._memory_history]
        
        return {
            'batches_processed': self._batch_counter,
            'avg_memory_usage_percent': sum(recent_usage) / len(recent_usage),
            'max_memory_usage_percent': max(recent_usage),
            'min_memory_usage_percent': min(recent_usage),
            'history_length': len(self._memory_history),
            'cleanup_callbacks_registered': len(self._cleanup_callbacks),
            'monitoring_active': self._monitor_thread is not None and self._monitor_thread.is_alive()
        }
    
    def recommend_batch_size(self, current_batch_size: int) -> int:
        """
        Recommend optimal batch size based on memory usage.
        
        Args:
            current_batch_size: Current batch size being used
            
        Returns:
            Recommended batch size
        """
        if not self._memory_history:
            return current_batch_size
        
        recent_usage = self._memory_history[-10:]  # Last 10 measurements
        avg_usage = sum(entry['limit_usage_percent'] for entry in recent_usage) / len(recent_usage)
        
        # Adjust batch size based on memory usage
        if avg_usage > self.critical_threshold * 100:
            # Reduce batch size significantly
            return max(1, int(current_batch_size * 0.5))
        elif avg_usage > self.warning_threshold * 100:
            # Reduce batch size moderately
            return max(1, int(current_batch_size * 0.8))
        elif avg_usage < 50:
            # Increase batch size if memory usage is low
            return min(current_batch_size * 2, 64)
        else:
            return current_batch_size
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
        
        # Final cleanup
        if exc_type is not None:
            print("Exception occurred, performing final cleanup...")
        
        self.trigger_cleanup(force=True)