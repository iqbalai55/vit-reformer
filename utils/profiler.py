import torch
import time
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from typing import List, Tuple, Optional
import psutil
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm

class ModelProfiler:
    def __init__(self, model: torch.nn.Module):
        """
        Initialize the profiler with a PyTorch model.

        Args:
            model: The PyTorch model to profile
        """
        self.model = model
        self.device = next(model.parameters()).device

    def profile_with_loader(self, train_loader: DataLoader, num_batches: Optional[int] = None) -> dict:
        """
        Profile model performance using actual training data.

        Args:
            train_loader: PyTorch DataLoader containing training data
            num_batches: Number of batches to profile (None for full epoch)

        Returns:
            Dictionary containing comprehensive profiling results
        """
        batch_times = []
        batch_memories = []
        batch_sizes = []
        batch_cpu_memories = []  # To track CPU memory usage during each batch
        cpu_memories = []  # To track total CPU memory usage during the profiling

        # Clear initial memory state
        torch.cuda.empty_cache()
        gc.collect()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        initial_ram = psutil.Process().memory_full_info().uss  # Initial CPU memory usage (USS)

        self.model.eval()
        max_memory = 0

        # Profile with actual batches
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(tqdm(train_loader, desc="Profiling")):
                if num_batches and i >= num_batches:
                    break

                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                batch_size = inputs.size(0)
                batch_sizes.append(batch_size)

                # Time the forward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.perf_counter()

                outputs = self.model(inputs)

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end = time.perf_counter()

                # Record metrics
                batch_times.append(end - start)
                current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                max_memory = max(max_memory, current_memory)
                batch_memories.append(current_memory)

                # Track CPU memory usage during the batch (USS)
                batch_cpu_memories.append(psutil.Process().memory_full_info().uss / 1024**2)  # Convert to MB

                # Track total CPU memory usage
                cpu_memories.append(psutil.virtual_memory().used / 1024**2)  # Convert to MB

                # Clear cache after each batch
                torch.cuda.empty_cache()

        # Get layer-wise profiling for a single batch
        sample_inputs, sample_targets = next(iter(train_loader))
        sample_inputs = sample_inputs.to(self.device)
        sample_targets = sample_targets.to(self.device)
        layer_profiles = self.profile_layers(sample_inputs)

        final_ram = psutil.Process().memory_full_info().uss  # Final CPU memory usage (USS)

        # Compile results
        results = {
            'batch_statistics': {
                'mean_batch_size': np.mean(batch_sizes),
                'total_samples_profiled': sum(batch_sizes),
                'num_batches_profiled': len(batch_times)
            },
            'timing_statistics': {
                'mean_batch_time_ms': np.mean(batch_times) * 1000,
                'std_batch_time_ms': np.std(batch_times) * 1000,
                'min_batch_time_ms': np.min(batch_times) * 1000,
                'max_batch_time_ms': np.max(batch_times) * 1000,
                'samples_per_second': np.mean(batch_sizes) / np.mean(batch_times)
            },
            'memory_statistics': {
                'initial_gpu_memory_mb': initial_memory / 1024**2,
                'peak_gpu_memory_mb': max_memory / 1024**2,
                'mean_batch_memory_mb': np.mean(batch_memories) / 1024**2,
                'initial_ram_mb': initial_ram / 1024**2,
                'final_ram_mb': final_ram / 1024**2,
                'ram_difference_mb': (final_ram - initial_ram) / 1024**2,
                'mean_cpu_memory_mb': np.mean(batch_cpu_memories),  # Mean CPU memory during profiling (USS)
                'mean_cpu_total_memory_mb': np.mean(cpu_memories),  # Mean total CPU memory during profiling
            },
            'layer_profiles': layer_profiles
        }

        return results

    def profile_layers(self, inputs: torch.Tensor) -> List[dict]:
        """
        Profile individual layers of the model using PyTorch Profiler.

        Args:
            inputs: Sample input tensor

        Returns:
            List of dictionaries containing layer-wise profiling information
        """
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        layer_profiles = []

        with profile(activities=activities, record_shapes=True) as prof:
            with record_function("model_forward"):
                _ = self.model(inputs)

        # Process profiler output
        for event in prof.key_averages():
            if event.key != "model_forward":
                layer_info = {
                    'name': event.key,
                    'cpu_time_total_ms': event.cpu_time_total / 1000000,  # Convert ns to ms
                    'self_cpu_time_total_ms': event.self_cpu_time_total / 1000000,  # Convert ns to ms
                    'cpu_memory_usage_mb': event.cpu_memory_usage / 1024**2 if hasattr(event, 'cpu_memory_usage') else 0,
                    'self_cpu_memory_usage_mb': event.self_cpu_memory_usage / 1024**2 if hasattr(event, 'self_cpu_memory_usage') else 0,
                    'input_shapes': event.input_shapes,
                }

                # Add device (CUDA) metrics if available
                if hasattr(event, 'device_time_total'):
                    layer_info['device_time_total_ms'] = event.device_time_total / 1000000
                if hasattr(event, 'self_device_time_total'):
                    layer_info['self_device_time_total_ms'] = event.self_device_time_total / 1000000
                if hasattr(event, 'device_memory_usage'):
                    layer_info['device_memory_usage_mb'] = event.device_memory_usage / 1024**2
                if hasattr(event, 'self_device_memory_usage'):
                    layer_info['self_device_memory_usage_mb'] = event.self_device_memory_usage / 1024**2

                layer_profiles.append(layer_info)

        return layer_profiles

    def profile_single_step_with_range(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        hyperparameter_name: str,
        range_values: list
    ) -> list:
        """
        Profile a single forward step of the model with varying values of a specified hyperparameter.

        Args:
            inputs: Input tensor for the model
            targets: Target tensor (unused but included for compatibility)
            hyperparameter_name: The name of the hyperparameter to vary
            range_values: A list of values to assign to the hyperparameter

        Returns:
            A list of dictionaries, each containing profiling results for a single step with a specific hyperparameter value
        """
        # Move data to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        results = []

        self.model.eval()
        for value in range_values:
            # Apply the hyperparameter value to the model
            if hasattr(self.model, hyperparameter_name):
                setattr(self.model, hyperparameter_name, value)
            else:
                raise AttributeError(f"Model does not have attribute '{hyperparameter_name}'")

            # Clear GPU memory and start profiling
            torch.cuda.empty_cache()
            gc.collect()
            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            # Time the forward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            with torch.no_grad():
                outputs = self.model(inputs)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()

            # Collect memory usage
            final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_used = final_memory - initial_memory

            # Append results for this hyperparameter value
            results.append({
                'hyperparameter_value': value,
                'forward_pass_time_ms': (end - start) * 1000,
                'gpu_memory_usage_mb': memory_used / 1024**2,
            })

        return results

def print_profiling_results(results: dict):
    """
    Print formatted profiling results.

    Args:
        results: Dictionary containing profiling results
    """
    print("\n=== Batch Statistics ===")
    for key, value in results['batch_statistics'].items():
        print(f"{key}: {value:.2f}")

    print("\n=== Timing Statistics ===")
    for key, value in results['timing_statistics'].items():
        print(f"{key}: {value:.2f}")

    print("\n=== Memory Statistics ===")
    for key, value in results['memory_statistics'].items():
        print(f"{key}: {value:.2f} MB")
