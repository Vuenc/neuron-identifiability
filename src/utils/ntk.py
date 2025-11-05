"""
Neural Tangent Kernel (NTK) computation utilities for tracking NTK evolution during training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
import gc
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict


def get_class_balanced_samples(
    dataset: Dataset,
    num_samples: int,
    seed: int = 42,
    classes: Optional[List[int]] = None
) -> List[int]:
    """
    Get class-balanced samples from a dataset.
    
    Args:
        dataset: PyTorch dataset
        num_samples: Total number of samples to return
        seed: Random seed for reproducibility
        classes: List of class indices. If None, will be inferred from dataset.
        
    Returns:
        List of dataset indices
    """
    # Get all labels from the dataset
    labels = []
    indices_by_class = defaultdict(list)
    
    for idx in range(len(dataset)):
        # Handle different dataset formats
        if hasattr(dataset, 'targets'):
            # Standard torchvision dataset
            label = dataset.targets[idx] if isinstance(dataset.targets, list) else dataset.targets[idx].item()
        elif hasattr(dataset, '__getitem__'):
            # Generic dataset - get label from first sample
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
        else:
            raise ValueError(f"Cannot extract labels from dataset type {type(dataset)}")
        
        labels.append(label)
        indices_by_class[label].append(idx)
    
    # If classes not specified, use all classes found in dataset
    if classes is None:
        classes = sorted(indices_by_class.keys())
    
    # Calculate samples per class
    num_classes = len(classes)
    samples_per_class = num_samples // num_classes
    remainder = num_samples % num_classes
    
    # Set random seed
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Sample indices from each class
    selected_indices = []
    for i, class_idx in enumerate(classes):
        if class_idx not in indices_by_class:
            continue
            
        class_indices = indices_by_class[class_idx]
        num_class_samples = samples_per_class + (1 if i < remainder else 0)
        num_class_samples = min(num_class_samples, len(class_indices))
        
        # Randomly sample from this class
        perm = torch.randperm(len(class_indices), generator=generator)
        selected_class_indices = [class_indices[perm[j].item()] for j in range(num_class_samples)]
        selected_indices.extend(selected_class_indices)
    
    # Shuffle the selected indices
    perm = torch.randperm(len(selected_indices), generator=generator)
    selected_indices = [selected_indices[perm[i].item()] for i in range(len(selected_indices))]
    
    return selected_indices[:num_samples]


def compute_ntk_matrix(
    model: nn.Module,
    data_points: torch.Tensor,
    device: str = 'cuda',
    output_dim: Optional[int] = None
) -> torch.Tensor:
    """
    Compute the Neural Tangent Kernel (NTK) matrix for a set of data points.
    
    The NTK matrix K[i, j] = <∇_θ f(x_i; θ), ∇_θ f(x_j; θ)>
    where f(x; θ) is the model output and θ are the parameters.
    
    Args:
        model: PyTorch model
        data_points: Tensor of shape (N, ...) where N is the number of data points
        device: Device to run computation on
        output_dim: Output dimension of the model. If None, will be inferred.
        
    Returns:
        NTK matrix of shape (N, N)
    """
    model.eval()
    model = model.to(device)
    data_points = data_points.to(device)
    
    N = data_points.shape[0]
    
    # Get output dimension
    if output_dim is None:
        with torch.no_grad():
            sample_output = model(data_points[0:1])
            if isinstance(sample_output, tuple):
                sample_output = sample_output[0]
            output_dim = sample_output.shape[-1]
    
    # Compute NTK matrix
    ntk_matrix = torch.zeros(N, N, device=device)
    
    for i in range(N):
        x_i = data_points[i:i+1]
        x_i.requires_grad_(True)
        
        # Compute output for x_i
        output_i = model(x_i)
        if isinstance(output_i, tuple):
            output_i = output_i[0]
        
        # Compute gradients for each output dimension
        for j in range(N):
            x_j = data_points[j:j+1]
            
            # For diagonal element (i == j), compute once
            if i == j:
                # Compute gradient w.r.t. parameters for all output dimensions
                grad_i = []
                for k in range(output_dim):
                    if output_i[0, k].requires_grad:
                        grad_k = torch.autograd.grad(
                            output_i[0, k],
                            [p for p in model.parameters() if p.requires_grad],
                            retain_graph=(k < output_dim - 1),
                            create_graph=False
                        )
                        # Flatten gradients
                        grad_k_flat = torch.cat([g.flatten() for g in grad_k])
                        grad_i.append(grad_k_flat)
                    else:
                        # If output doesn't require grad, use zeros
                        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        grad_k_flat = torch.zeros(total_params, device=device)
                        grad_i.append(grad_k_flat)
                
                # Stack gradients for all output dimensions
                if len(grad_i) > 0:
                    grad_i_all = torch.stack(grad_i)  # (output_dim, num_params)
                    # For NTK, we sum over output dimensions (can be modified for multi-output)
                    grad_i_sum = grad_i_all.sum(dim=0)  # (num_params,)
                    ntk_ij = torch.dot(grad_i_sum, grad_i_sum)
                else:
                    ntk_ij = torch.tensor(0.0, device=device)
                
                ntk_matrix[i, j] = ntk_ij.item()
            else:
                # For off-diagonal, need gradients for both x_i and x_j
                x_j.requires_grad_(True)
                output_j = model(x_j)
                if isinstance(output_j, tuple):
                    output_j = output_j[0]
                
                grad_i = []
                for k in range(output_dim):
                    if output_i[0, k].requires_grad:
                        grad_k_i = torch.autograd.grad(
                            output_i[0, k],
                            [p for p in model.parameters() if p.requires_grad],
                            retain_graph=(k < output_dim - 1),
                            create_graph=False
                        )
                        grad_k_i_flat = torch.cat([g.flatten() for g in grad_k_i])
                        grad_i.append(grad_k_i_flat)
                    else:
                        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        grad_k_i_flat = torch.zeros(total_params, device=device)
                        grad_i.append(grad_k_i_flat)
                
                grad_j = []
                for k in range(output_dim):
                    if output_j[0, k].requires_grad:
                        grad_k_j = torch.autograd.grad(
                            output_j[0, k],
                            [p for p in model.parameters() if p.requires_grad],
                            retain_graph=(k < output_dim - 1),
                            create_graph=False
                        )
                        grad_k_j_flat = torch.cat([g.flatten() for g in grad_k_j])
                        grad_j.append(grad_k_j_flat)
                    else:
                        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        grad_k_j_flat = torch.zeros(total_params, device=device)
                        grad_j.append(grad_k_j_flat)
                
                # Compute dot product
                if len(grad_i) > 0 and len(grad_j) > 0:
                    grad_i_all = torch.stack(grad_i).sum(dim=0)
                    grad_j_all = torch.stack(grad_j).sum(dim=0)
                    ntk_ij = torch.dot(grad_i_all, grad_j_all)
                else:
                    ntk_ij = torch.tensor(0.0, device=device)
                
                ntk_matrix[i, j] = ntk_ij.item()
                x_j.requires_grad_(False)
        
        x_i.requires_grad_(False)
    
    # Make symmetric (since K(x_i, x_j) = K(x_j, x_i))
    ntk_matrix = (ntk_matrix + ntk_matrix.T) / 2
    
    return ntk_matrix


def compute_ntk_matrix_efficient(
    model: nn.Module,
    data_points: torch.Tensor,
    device: str = 'cuda',
    output_dim: Optional[int] = None,
    batch_size: int = 1,
    chunk_size: int = 10
) -> torch.Tensor:
    """
    Memory-efficient NTK computation using chunked gradient computation.
    Computes NTK incrementally to avoid storing all gradients at once.
    
    Args:
        model: PyTorch model
        data_points: Tensor of shape (N, ...) where N is the number of data points
        device: Device to run computation on
        output_dim: Output dimension of the model. If None, will be inferred.
        batch_size: Batch size for processing individual samples (typically 1)
        chunk_size: Number of samples to process at once for NTK computation (memory vs speed tradeoff)
        
    Returns:
        NTK matrix of shape (N, N)
    """
    model.eval()
    model = model.to(device)
    data_points = data_points.to(device)
    
    N = data_points.shape[0]
    
    # Get output dimension
    if output_dim is None:
        with torch.no_grad():
            sample_output = model(data_points[0:1])
            if isinstance(sample_output, tuple):
                sample_output = sample_output[0]
            output_dim = sample_output.shape[-1]
    
    # Get all parameters
    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)
    
    # Initialize NTK matrix on CPU to save GPU memory
    ntk_matrix = torch.zeros(N, N, device='cpu')
    
    def compute_gradients_for_sample(x):
        """Compute gradients for a single sample, returns (output_dim, total_params)"""
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]
        
        grad_per_output = []
        for k in range(output_dim):
            grad_k = torch.autograd.grad(
                output[0, k],
                params,
                retain_graph=(k < output_dim - 1),
                create_graph=False
            )
            grad_k_flat = torch.cat([g.flatten() for g in grad_k])
            grad_per_output.append(grad_k_flat)
        
        # Stack to shape (output_dim, total_params)
        return torch.stack(grad_per_output)  # (output_dim, total_params)
    
    # Process data in chunks to avoid storing all gradients
    # Use a small cache for recent chunks to avoid recomputing
    num_chunks = (N + chunk_size - 1) // chunk_size
    chunk_cache = {}  # Maps chunk_idx -> gradients on CPU
    max_cache_size = 3  # Keep last 3 chunks in cache
    
    for chunk_i in range(num_chunks):
        start_i = chunk_i * chunk_size
        end_i = min(start_i + chunk_size, N)
        chunk_indices = list(range(start_i, end_i))
        
        # Compute gradients for this chunk
        chunk_grads = []
        for idx in chunk_indices:
            x = data_points[idx:idx+1]
            grads = compute_gradients_for_sample(x)
            # Move to CPU immediately to save GPU memory
            chunk_grads.append(grads.cpu())
        
        # Stack chunk gradients on CPU: (chunk_size, output_dim, total_params)
        chunk_grads = torch.stack(chunk_grads)  # On CPU
        
        # Add to cache
        chunk_cache[chunk_i] = chunk_grads
        # Evict oldest chunks if cache is too large
        if len(chunk_cache) > max_cache_size:
            oldest_key = min(chunk_cache.keys())
            del chunk_cache[oldest_key]
        
        # Process all previous chunks to compute cross-chunk contributions
        # and within-chunk contributions (we only compute upper triangle + diagonal)
        for prev_chunk_j in range(chunk_i + 1):
            start_j = prev_chunk_j * chunk_size
            end_j = min(start_j + chunk_size, N)
            
            # Get gradients for previous chunk (from cache or recompute)
            if prev_chunk_j in chunk_cache:
                prev_chunk_grads = chunk_cache[prev_chunk_j]
            else:
                # Need to recompute (cache miss)
                prev_chunk_grads = []
                prev_chunk_indices = list(range(start_j, end_j))
                for idx in prev_chunk_indices:
                    x = data_points[idx:idx+1]
                    grads = compute_gradients_for_sample(x)
                    prev_chunk_grads.append(grads.cpu())
                prev_chunk_grads = torch.stack(prev_chunk_grads)  # On CPU
            
            # Compute NTK contribution for this pair of chunks
            # For each output dimension k, compute pairwise gradient products
            # Move chunks to GPU temporarily for computation
            chunk_grads_gpu = chunk_grads.to(device)
            prev_chunk_grads_gpu = prev_chunk_grads.to(device)
            
            for k in range(output_dim):
                grads_i_k = chunk_grads_gpu[:, k, :]  # (chunk_size, total_params)
                grads_j_k = prev_chunk_grads_gpu[:, k, :]  # (prev_chunk_size, total_params)
                
                # Compute pairwise products: (chunk_size, prev_chunk_size)
                chunk_ntk = torch.mm(grads_i_k, grads_j_k.T)  # (chunk_size, prev_chunk_size)
                
                # Move to CPU and add to global NTK matrix
                chunk_ntk_cpu = chunk_ntk.cpu()
                ntk_matrix[start_i:end_i, start_j:end_j] += chunk_ntk_cpu
                
                # If not on diagonal, also update symmetric part
                if prev_chunk_j < chunk_i:
                    ntk_matrix[start_j:end_j, start_i:end_i] += chunk_ntk_cpu.T
                
                # Clean up GPU memory
                del chunk_ntk, chunk_ntk_cpu
            
            # Clean up temporary GPU copies
            del chunk_grads_gpu, prev_chunk_grads_gpu
        
        # Clean up current chunk gradients from GPU if they were there
        # (they're in cache on CPU now, so we're good)
        
        # Force garbage collection and clear GPU cache periodically
        if (chunk_i + 1) % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Move final matrix to device if needed
    ntk_matrix = ntk_matrix.to(device)
    
    # Make symmetric (should already be symmetric, but ensure numerical stability)
    ntk_matrix = (ntk_matrix + ntk_matrix.T) / 2
    
    return ntk_matrix


def compute_cka(gram1: torch.Tensor, gram2: torch.Tensor) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two gram matrices.
    
    CKA is a similarity measure between two kernel matrices that is invariant to 
    orthogonal transformations and isotropic scaling.
    
    Args:
        gram1: First gram matrix (NTK matrix) of shape (N, N)
        gram2: Second gram matrix (NTK matrix) of shape (N, N)
        
    Returns:
        CKA value between 0 and 1 (1 = identical, 0 = orthogonal)
    """
    assert gram1.shape == gram2.shape, f"Gram matrices must have same shape, got {gram1.shape} and {gram2.shape}"
    assert gram1.shape[0] == gram1.shape[1], "Gram matrices must be square"
    
    N = gram1.shape[0]
    
    # Convert to numpy for computation
    if isinstance(gram1, torch.Tensor):
        gram1_np = gram1.cpu().numpy()
        gram2_np = gram2.cpu().numpy()
    else:
        gram1_np = gram1
        gram2_np = gram2
    
    # Center both kernel matrices using centering matrix H = I - (1/N) * ones
    # K_centered = H @ K @ H where H = I - (1/N) * ones
    # For efficient computation, we can center directly:
    # K_centered = K - mean_row - mean_col + mean_all
    # Or use matrix multiplication with H
    
    # Build centering matrix H = I - (1/N) * ones
    H = np.eye(N) - (1.0 / N) * np.ones((N, N))
    
    # Center the matrices: K_centered = H @ K @ H
    gram1_centered = H @ gram1_np @ H
    gram2_centered = H @ gram2_np @ H
    
    # Compute CKA: CKA(K1, K2) = ||K1_centered @ K2_centered||_F^2 / (||K1_centered||_F^2 ||K2_centered||_F^2)
    # For symmetric matrices: ||K||_F^2 = trace(K @ K) = trace(K^T @ K)
    # Numerator: ||K1_centered @ K2_centered||_F^2 = trace((K1_centered @ K2_centered)^T @ (K1_centered @ K2_centered))
    #            = trace(K2_centered @ K1_centered @ K1_centered @ K2_centered)
    # For efficiency, we use: trace(K1_centered @ K2_centered)^2
    # Denominator: ||K1_centered||_F^2 = trace(K1_centered @ K1_centered)
    #               ||K2_centered||_F^2 = trace(K2_centered @ K2_centered)
    
    # Since matrices are symmetric after centering, trace(K @ K) = trace(K^T @ K)
    numerator = np.trace(gram1_centered @ gram2_centered) ** 2
    denom1 = np.trace(gram1_centered @ gram1_centered)
    denom2 = np.trace(gram2_centered @ gram2_centered)
    
    if denom1 > 0 and denom2 > 0:
        cka = numerator / (denom1 * denom2)
    else:
        cka = 0.0
    
    return float(cka)


def compute_ntk_stats(ntk_matrix: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics from the NTK matrix.
    
    Args:
        ntk_matrix: NTK matrix of shape (N, N)
        
    Returns:
        Dictionary containing NTK statistics
    """
    ntk_np = ntk_matrix.cpu().numpy()
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(ntk_np)  # Already sorted in ascending order
    
    stats = {
        'trace': float(np.trace(ntk_np)),
        'frobenius_norm': float(np.linalg.norm(ntk_np, 'fro')),
        'max_eigenvalue': float(np.max(eigenvalues)),
        'min_eigenvalue': float(np.min(eigenvalues)),
        'mean_eigenvalue': float(np.mean(eigenvalues)),
        'median_eigenvalue': float(np.median(eigenvalues)),
        'condition_number': float(np.max(eigenvalues) / (np.min(eigenvalues) + 1e-10)),
        'effective_rank': float(np.sum(eigenvalues) / (np.max(eigenvalues) + 1e-10)),
    }
    
    return stats


def compute_linearization_agreement(
    init_model: nn.Module,
    current_model: nn.Module,
    data_points: torch.Tensor,
    device: str = 'cuda',
    output_dim: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute agreement between the actual function f_theta and its linearization around initialization.
    
    Linearization: f_linearized = f_theta_0 + J_theta_0 @ (theta - theta_0)
    where:
    - f_theta_0 is the output at initialization
    - J_theta_0 is the Jacobian at initialization (gradients w.r.t. parameters)
    - theta - theta_0 is the parameter change
    
    Args:
        init_model: Model at initialization (theta_0)
        current_model: Current model (theta)
        data_points: Tensor of shape (N, ...) with data points to evaluate on
        device: Device to run computation on
        output_dim: Output dimension. If None, will be inferred.
        
    Returns:
        Dictionary with agreement metrics:
        - mse: Mean squared error between f_theta and f_linearized
        - relative_error: Relative error ||f_theta - f_linearized|| / ||f_theta||
        - cosine_sim: Cosine similarity between f_theta and f_linearized
        - max_error: Maximum element-wise error
        - mean_abs_error: Mean absolute error
    """
    init_model.eval()
    current_model.eval()
    init_model = init_model.to(device)
    current_model = current_model.to(device)
    data_points = data_points.to(device)
    
    N = data_points.shape[0]
    
    # Get output dimension
    if output_dim is None:
        with torch.no_grad():
            sample_output = init_model(data_points[0:1])
            if isinstance(sample_output, tuple):
                sample_output = sample_output[0]
            output_dim = sample_output.shape[-1]
    
    # Get all parameters
    init_params = [p for p in init_model.parameters() if p.requires_grad]
    current_params = [p for p in current_model.parameters() if p.requires_grad]
    
    # Compute parameter change: (theta - theta_0)
    param_diff = []
    for curr_p, init_p in zip(current_params, init_params):
        param_diff.append((curr_p - init_p).flatten())
    param_diff = torch.cat(param_diff)  # (total_params,)
    
    # Compute f_theta_0 (outputs at initialization)
    with torch.no_grad():
        init_outputs = []
        for i in range(N):
            x = data_points[i:i+1]
            out = init_model(x)
            if isinstance(out, tuple):
                out = out[0]
            init_outputs.append(out[0])  # (output_dim,)
        f_theta_0 = torch.stack(init_outputs)  # (N, output_dim)
    
    # Compute J_theta_0 (Jacobian at initialization) and linearized predictions
    # Memory-efficient: compute gradients per sample and immediately compute linearized output
    # Instead of storing all Jacobians (N, output_dim, total_params) which would be huge
    f_linearized = f_theta_0.clone()
    
    # Process in chunks to avoid storing all Jacobians at once
    chunk_size = 10  # Process 10 samples at a time to limit memory usage
    
    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        chunk_indices = range(chunk_start, chunk_end)
        
        # Compute Jacobian for this chunk
        jacobian_chunk = []
        for i in chunk_indices:
            x = data_points[i:i+1]
            output = init_model(x)
            if isinstance(output, tuple):
                output = output[0]
            
            grad_per_output = []
            for k in range(output_dim):
                grad_k = torch.autograd.grad(
                    output[0, k],
                    init_params,
                    retain_graph=(k < output_dim - 1),
                    create_graph=False
                )
                grad_k_flat = torch.cat([g.flatten() for g in grad_k])
                grad_per_output.append(grad_k_flat)
            
            # Stack to shape (output_dim, total_params)
            grad_per_output = torch.stack(grad_per_output)  # (output_dim, total_params)
            jacobian_chunk.append(grad_per_output)
        
        # Compute linearized predictions for this chunk immediately
        jacobian_chunk = torch.stack(jacobian_chunk)  # (chunk_size, output_dim, total_params)
        for idx, i in enumerate(chunk_indices):
            # jacobian_chunk[idx] is (output_dim, total_params), param_diff is (total_params,)
            # Result is (output_dim,)
            f_linearized[i] += jacobian_chunk[idx] @ param_diff
        
        # Clean up chunk Jacobians immediately
        del jacobian_chunk
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Compute f_theta (outputs at current model)
    with torch.no_grad():
        current_outputs = []
        for i in range(N):
            x = data_points[i:i+1]
            out = current_model(x)
            if isinstance(out, tuple):
                out = out[0]
            current_outputs.append(out[0])  # (output_dim,)
        f_theta = torch.stack(current_outputs)  # (N, output_dim)
    
    # Compute agreement metrics
    error = f_theta - f_linearized  # (N, output_dim)
    
    # MSE
    mse = torch.mean(error ** 2).item()
    
    # Mean absolute error
    mean_abs_error = torch.mean(torch.abs(error)).item()
    
    # Maximum element-wise error
    max_error = torch.max(torch.abs(error)).item()
    
    # Relative error: ||error|| / ||f_theta||
    f_theta_norm = torch.norm(f_theta).item()
    if f_theta_norm > 1e-10:
        relative_error = torch.norm(error).item() / f_theta_norm
    else:
        relative_error = float('inf') if torch.norm(error).item() > 1e-10 else 0.0
    
    # Cosine similarity (flatten both to vectors)
    f_theta_flat = f_theta.flatten()
    f_linearized_flat = f_linearized.flatten()
    norm_f_theta = torch.norm(f_theta_flat)
    norm_f_linearized = torch.norm(f_linearized_flat)
    
    if norm_f_theta > 1e-10 and norm_f_linearized > 1e-10:
        cosine_sim = torch.dot(f_theta_flat, f_linearized_flat).item() / (norm_f_theta * norm_f_linearized)
    else:
        cosine_sim = 0.0
    
    # For classification tasks: accuracy agreement
    # Check if predictions (argmax) agree
    if output_dim > 1:  # Classification task
        pred_theta = torch.argmax(f_theta, dim=1)
        pred_linearized = torch.argmax(f_linearized, dim=1)
        accuracy_agreement = (pred_theta == pred_linearized).float().mean().item()
    else:
        accuracy_agreement = None
    
    results = {
        'mse': mse,
        'mean_abs_error': mean_abs_error,
        'max_error': max_error,
        'relative_error': relative_error,
        'cosine_sim': cosine_sim,
    }
    
    if accuracy_agreement is not None:
        results['accuracy_agreement'] = accuracy_agreement
    
    return results


def sample_data_points(
    data_loader: DataLoader,
    num_samples: int,
    class_balanced: bool = True,
    seed: int = 42
) -> torch.Tensor:
    """
    Sample data points from a DataLoader without computing NTK.
    This is a lightweight function to get selected_data for linearization agreement.
    
    Args:
        data_loader: DataLoader containing the dataset
        num_samples: Number of data points to sample
        class_balanced: Whether to use class-balanced sampling
        seed: Random seed
        
    Returns:
        Selected data points tensor of shape (num_samples, ...)
    """
    # Collect all data points
    all_data = []
    all_labels = []
    
    for data, labels in data_loader:
        all_data.append(data)
        all_labels.append(labels)
    
    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Select samples
    if class_balanced and len(torch.unique(all_labels)) > 1:
        # Get unique classes
        unique_classes = torch.unique(all_labels).tolist()
        
        # Sample balanced indices
        indices_by_class = defaultdict(list)
        for idx, label in enumerate(all_labels):
            indices_by_class[label.item()].append(idx)
        
        samples_per_class = num_samples // len(unique_classes)
        remainder = num_samples % len(unique_classes)
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        selected_indices = []
        for i, class_idx in enumerate(unique_classes):
            class_indices = indices_by_class[class_idx]
            num_class_samples = samples_per_class + (1 if i < remainder else 0)
            num_class_samples = min(num_class_samples, len(class_indices))
            
            perm = torch.randperm(len(class_indices), generator=generator)
            selected_class_indices = [class_indices[perm[j].item()] for j in range(num_class_samples)]
            selected_indices.extend(selected_class_indices)
        
        # Shuffle
        perm = torch.randperm(len(selected_indices), generator=generator)
        selected_indices = [selected_indices[perm[i].item()] for i in range(len(selected_indices))]
    else:
        # Random sampling
        generator = torch.Generator()
        generator.manual_seed(seed)
        total_samples = len(all_data)
        num_samples = min(num_samples, total_samples)
        perm = torch.randperm(total_samples, generator=generator)
        selected_indices = perm[:num_samples].tolist()
    
    # Get selected data points
    selected_data = all_data[selected_indices]
    
    return selected_data


def compute_ntk_from_dataloader(
    model: nn.Module,
    data_loader: DataLoader,
    num_samples: int,
    class_balanced: bool = True,
    seed: int = 42,
    device: str = 'cuda',
    return_selected_data: bool = False
) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor]]:
    """
    Compute NTK matrix from a DataLoader with optional class-balanced sampling.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader containing the dataset
        num_samples: Number of data points to use for NTK computation
        class_balanced: Whether to use class-balanced sampling
        seed: Random seed
        device: Device to run computation on
        return_selected_data: If True, also return the selected data points
        
    Returns:
        Tuple of (NTK matrix, statistics dictionary, selected_data)
        If return_selected_data=False, selected_data will be None
    """
    # Collect all data points
    all_data = []
    all_labels = []
    
    for data, labels in data_loader:
        all_data.append(data)
        all_labels.append(labels)
    
    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Select samples
    if class_balanced and len(torch.unique(all_labels)) > 1:
        # Get unique classes
        unique_classes = torch.unique(all_labels).tolist()
        
        # Sample balanced indices
        indices_by_class = defaultdict(list)
        for idx, label in enumerate(all_labels):
            indices_by_class[label.item()].append(idx)
        
        samples_per_class = num_samples // len(unique_classes)
        remainder = num_samples % len(unique_classes)
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        selected_indices = []
        for i, class_idx in enumerate(unique_classes):
            class_indices = indices_by_class[class_idx]
            num_class_samples = samples_per_class + (1 if i < remainder else 0)
            num_class_samples = min(num_class_samples, len(class_indices))
            
            perm = torch.randperm(len(class_indices), generator=generator)
            selected_class_indices = [class_indices[perm[j].item()] for j in range(num_class_samples)]
            selected_indices.extend(selected_class_indices)
        
        # Shuffle
        perm = torch.randperm(len(selected_indices), generator=generator)
        selected_indices = [selected_indices[perm[i].item()] for i in range(len(selected_indices))]
    else:
        # Random sampling
        generator = torch.Generator()
        generator.manual_seed(seed)
        total_samples = len(all_data)
        num_samples = min(num_samples, total_samples)
        perm = torch.randperm(total_samples, generator=generator)
        selected_indices = perm[:num_samples].tolist()
    
    # Get selected data points
    selected_data = all_data[selected_indices]
    
    # Compute NTK matrix
    ntk_matrix = compute_ntk_matrix_efficient(model, selected_data, device=device)
    
    # Compute statistics
    stats = compute_ntk_stats(ntk_matrix)
    
    if return_selected_data:
        return ntk_matrix, stats, selected_data
    else:
        return ntk_matrix, stats, None


def compute_ntk_gnn(
    model: nn.Module,
    data,
    indices: torch.Tensor,
    device: str = 'cuda',
    output_dim: Optional[int] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute NTK matrix for GNN models.
    
    Args:
        model: GNN model
        data: Graph data object
        indices: Node indices to use for NTK computation
        device: Device to run computation on
        output_dim: Output dimension. If None, will be inferred.
        
    Returns:
        Tuple of (NTK matrix, statistics dictionary)
    """
    model.eval()
    model = model.to(device)
    
    # Get output dimension
    if output_dim is None:
        with torch.no_grad():
            sample_output = model(data.x, data.adj_t)
            if isinstance(sample_output, tuple):
                sample_output = sample_output[0]
            output_dim = sample_output.shape[-1]
    
    N = len(indices)
    indices = indices.to(device)
    
    # Get all parameters
    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)
    
    # Compute gradient for each node
    gradients = []
    
    for i in range(N):
        node_idx = indices[i:i+1]
        
        # Forward pass for this node
        output = model(data.x, data.adj_t)
        if isinstance(output, tuple):
            output = output[0]
        
        node_output = output[node_idx]  # (1, output_dim)
        
        # Compute gradient for each output dimension separately
        # For multi-output NTK: K[i,j] = Σ_k ∇_θ f_k(x_i; θ)^T ∇_θ f_k(x_j; θ)
        grad_per_output = []
        for k in range(output_dim):
            grad_k = torch.autograd.grad(
                node_output[0, k],
                params,
                retain_graph=(k < output_dim - 1),
                create_graph=False
            )
            grad_k_flat = torch.cat([g.flatten() for g in grad_k])
            grad_per_output.append(grad_k_flat)
        
        # Stack to shape (output_dim, total_params)
        grad_per_output = torch.stack(grad_per_output)
        gradients.append(grad_per_output)
    
    # Stack all gradients: shape (N, output_dim, total_params)
    gradients = torch.stack(gradients)  # (N, output_dim, total_params)
    
    # Compute NTK matrix: K[i,j] = Σ_k ∇_θ f_k(x_i; θ)^T ∇_θ f_k(x_j; θ)
    ntk_matrix = torch.zeros(N, N, device=device)
    for k in range(output_dim):
        # Extract gradients for output dimension k: (N, total_params)
        grads_k = gradients[:, k, :]  # (N, total_params)
        # Add pairwise products for this output dimension
        ntk_matrix += torch.mm(grads_k, grads_k.T)  # (N, N)
    
    # Compute statistics
    stats = compute_ntk_stats(ntk_matrix)
    
    return ntk_matrix, stats

