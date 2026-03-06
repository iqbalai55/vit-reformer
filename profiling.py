from reformer.vir_pytorch import ViR
from vit_pytorch import ViT
import matplotlib.pyplot as plt
import numpy as np
import csv
from utils.datasets import get_lpfw_dataloaders
import os
from utils.profiler import ModelProfiler
import torch 

# Create results directory if it doesn't exist
os.makedirs('hasil/profiling_results', exist_ok=True)

# Updated CSV headers to include CPU metrics
csv_headers = [
    "Model", "Batch Size", "Patch Size", 
    "Mean Batch Time (ms)", "Standard Deviation (ms)", 
    "Throughput (samples/second)", 
    "Peak GPU Memory (MB)", "Mean Batch Memory (MB)",
    "Mean CPU Memory (MB)", "Peak CPU Memory (MB)",
    "CPU Memory Change (MB)", "Mean Total CPU Memory (MB)"
]

with open('hasil/profiling_results/performance_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)

# Define parameter ranges
batch_sizes = range(1, 8)
patch_sizes = [4, 8, 16, 32]

def create_models(patch_size, num_classes):
    """Create both models with given patch size and move them to the device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_reformer = ViR(
            img_size=224,
            patch_size=patch_size,
            in_channels=3,  # Corrected from 1000 to 3 (input channels for RGB images)
            num_classes=num_classes,
            dim=256,
            depth=12,
            heads=8,
            bucket_size=5,
            n_hashes=1,
            ff_mult=4,  # Added missing argument
            lsh_dropout=0.1,  # Corrected from dropout to lsh_dropout
            ff_dropout=0.1,  # Added missing argument
            emb_dropout=0.1,
            use_rezero=False  # Added missing argument
        )
    
    model_vit = ViT(
        image_size=224,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=256,
        depth=12,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    model_reformer.to(device)
    model_vit.to(device)
    return model_reformer, model_vit

def save_results(model_name, batch_size, patch_size, results):
    """Save profiling results to CSV."""
    with open('hasil/profiling_results/performance_metrics.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            model_name,
            batch_size,
            patch_size,
            results['timing_statistics']['mean_batch_time_ms'],
            results['timing_statistics']['std_batch_time_ms'],
            results['timing_statistics']['samples_per_second'],
            results['memory_statistics']['peak_gpu_memory_mb'],
            results['memory_statistics']['mean_batch_memory_mb'],
            results['memory_statistics']['mean_cpu_memory_mb'],
            max(results['memory_statistics']['mean_cpu_memory_mb'], 
                results['memory_statistics']['final_ram_mb']),
            results['memory_statistics']['ram_difference_mb'],
            results['memory_statistics']['mean_cpu_total_memory_mb']
        ])

# Updated plot data structure to include CPU metrics
plot_data = {
    'reformer': {
        'batch_times': [], 'throughput': [], 
        'gpu_memory': [], 'cpu_memory': [], 
        'total_cpu_memory': []
    },
    'vit': {
        'batch_times': [], 'throughput': [], 
        'gpu_memory': [], 'cpu_memory': [], 
        'total_cpu_memory': []
    }
}

# Main profiling loop
for patch_size in patch_sizes:
    print(f"\nProfiling with patch size: {patch_size}")
    
    for batch_size in batch_sizes:
        print(f"Processing batch size: {batch_size}")
        
        train_loader, test_loader = get_lpfw_dataloaders(batch_size)
        num_classes = len(train_loader.dataset.dataset.class_to_idx)
        
        model_reformer, model_vit = create_models(patch_size, num_classes)
        
        # Profile both models
        for model_name, model in [('reformer', model_reformer), ('vit', model_vit)]:
            profiler = ModelProfiler(model)
            results = profiler.profile_with_loader(train_loader, num_batches=1)
            save_results(model_name.capitalize(), batch_size, patch_size, results)
            
            # Store results for plotting
            plot_data[model_name]['batch_times'].append({
                'patch_size': patch_size,
                'batch_size': batch_size,
                'value': results['timing_statistics']['mean_batch_time_ms']
            })
            plot_data[model_name]['throughput'].append({
                'patch_size': patch_size,
                'batch_size': batch_size,
                'value': results['timing_statistics']['samples_per_second']
            })
            plot_data[model_name]['gpu_memory'].append({
                'patch_size': patch_size,
                'batch_size': batch_size,
                'value': results['memory_statistics']['peak_gpu_memory_mb']
            })
            plot_data[model_name]['cpu_memory'].append({
                'patch_size': patch_size,
                'batch_size': batch_size,
                'value': results['memory_statistics']['mean_cpu_memory_mb']
            })
            plot_data[model_name]['total_cpu_memory'].append({
                'patch_size': patch_size,
                'batch_size': batch_size,
                'value': results['memory_statistics']['mean_cpu_total_memory_mb']
            })

# Create plots with CPU metrics
fig, axs = plt.subplots(5, len(patch_sizes), figsize=(20, 25))
metrics = ['batch_times', 'throughput', 'gpu_memory', 'cpu_memory', 'total_cpu_memory']
titles = ['Batch Time', 'Throughput', 'GPU Memory Usage', 'CPU Memory Usage', 'Total CPU Memory Usage']
ylabels = ['Time (ms)', 'Samples/Second', 'GPU Memory (MB)', 'CPU Memory (MB)', 'Total CPU Memory (MB)']

for i, metric in enumerate(metrics):
    for j, patch_size in enumerate(patch_sizes):
        # Filter data for current patch size
        reformer_data = [x for x in plot_data['reformer'][metric] if x['patch_size'] == patch_size]
        vit_data = [x for x in plot_data['vit'][metric] if x['patch_size'] == patch_size]
        
        # Plot
        axs[i, j].plot(
            [x['batch_size'] for x in reformer_data],
            [x['value'] for x in reformer_data],
            label='Reformer',
            marker='o'
        )
        axs[i, j].plot(
            [x['batch_size'] for x in vit_data],
            [x['value'] for x in vit_data],
            label='ViT',
            marker='s'
        )
        
        axs[i, j].set_title(f'{titles[i]} (Patch Size {patch_size})')
        axs[i, j].set_xlabel('Batch Size')
        axs[i, j].set_ylabel(ylabels[i])
        axs[i, j].legend()
        axs[i, j].grid(True)

plt.tight_layout()
plt.savefig('hasil/profiling_results/performance_comparison.png')
plt.close()