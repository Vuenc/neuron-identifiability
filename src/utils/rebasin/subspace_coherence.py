import collections
from dataclasses import dataclass
import math
from typing import DefaultDict, Dict, List, Tuple

import torch
from torch.nn import Linear

from src.models.mlp import SparseLinear, NoiseLinear
import src.utils.record_activations
from src.utils.record_activations import MODEL_INPUT_RECORDING_POINT, MODEL_OUTPUT_RECORDING_POINT, RecordInput

from .common import PermutationSpec, ActivationRecordingPoint, LayerName

@dataclass
class SubspaceCoherenceResult:
  subspace_coherence: float
  explained_variance_ratio: float
  subspace_dimension: int
  full_dimension: int
  anisotropy_operator_norms: List[float] | None
  mean_anisotropy_operator_norm: float | None
  anisotropy_bounds: Dict[float, float] | None

def estimate_subspace_basis_at_explained_variance(data: torch.Tensor, target_explained_variance_ratio: float) -> Tuple[torch.Tensor, float]:
  data = data - data.mean(dim=0)
  _, S, UT = torch.linalg.svd(data, full_matrices=False)
  explained_variance_ratio_cum = torch.cumsum((S**2) / (S**2).sum(), 0)

  # torch.searchsorted returns an index k such that the explained_variance_ratio_cum[k-1] < target_explained_variance_ratio <= explained_variance_ratio_cum[k]
  #   We therefore need to include basis vectors up to k (inclusive), i.e. the range `:k+1`
  # Edge case: if target_explained_variance_ratio > explained_variance_ratio_cum[-1], k = len(explained_variance_ratio_cum). This can happen if
  #   target_explained_variance_ratio > 1 i specified, or e.g. target_explained_variance_ratio = 1, but explained_variance_ratio_cum[-1] < 1 due to
  #   numerical errors. To avoid, we use k = min(k, len(explained_variance_ratio_cum)-1)
  k: int = torch.searchsorted(
      explained_variance_ratio_cum,
      target_explained_variance_ratio
  ).item() # type: ignore
  k = min(k, len(explained_variance_ratio_cum)-1)

  U = UT.T[:, :k+1]
  explained_variance_ratio= explained_variance_ratio_cum[k].item()
  return U, explained_variance_ratio

def evaluate_anisotropy(subspace_basis, layer: SparseLinear | Linear | NoiseLinear | None, mask_distribution_mean: float):
  # Subspace basis: U (in the notation of the paper)
  # Diagonal operator D (in the notation of the paper)
  # TODO check if the dimensions are correct, or if it needs to be transposed (check on a layer where input dimension != output dimension)
  if not (isinstance(layer, SparseLinear) or isinstance(layer, Linear) or isinstance(layer, NoiseLinear)):
    raise ValueError("Argument layer must be of type Linear, SparseLinear, or NoiseLinear")
  diagonal_operator: torch.Tensor = (
    layer.mask
    if isinstance(layer, SparseLinear)
    else torch.ones_like(layer.weight)
  )  # type: ignore

  # Projected diagonal operators M_i (in the notation of the paper)
  #   - Paper: compute U.T @ diag(d_i), where d_i = D[i-1].
  #   - Here: compute for all neurons i, where the dimensions are ("neurons", "subspace dimension", "full space dimension")
  projected_diagonal_operators = torch.einsum("dk,nd->nkd", subspace_basis, diagonal_operator)

  # Gram matrix S_i (in the notation of the paper)
  #    - Here: compute for all neurons i, where the dimensions are ("neurons", "subspace dimension", "subspace dimension")
  gram_matrices = torch.einsum("nkd,nKd->nkK", projected_diagonal_operators, projected_diagonal_operators)

  difference_from_identity = gram_matrices - torch.eye(gram_matrices.shape[1], device=gram_matrices.device)*mask_distribution_mean
  operator_norms = torch.linalg.matrix_norm(difference_from_identity, ord=2)

  return operator_norms

def compute_subspace_coherence_and_anisotropy(
    data: torch.Tensor,
    target_explained_variance_ratio: float,
    layer: SparseLinear | Linear | NoiseLinear | None,
    mask_distribution_mean: float | None,
    mask_distribution_ess_sup: float | None,
    mask_distribution_std_dev: float | None,
    anisotropy_bound_violation_probabilities = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
) -> SubspaceCoherenceResult:
  subspace_basis, explained_variance_ratio = estimate_subspace_basis_at_explained_variance(
    data, target_explained_variance_ratio)
  subspace_coherence = (subspace_basis**2).sum(dim=1).max().item()

  if layer is not None:
    if mask_distribution_mean is None or mask_distribution_ess_sup is None or mask_distribution_std_dev is None:
      raise ValueError("Parameters mask_distribution_mean, mask_distribution_ess_sup, mask_distribution_std_dev must be passed to compute anisotropy bounds!")
    anisotropy_operator_norms = evaluate_anisotropy(subspace_basis=subspace_basis, layer=layer, mask_distribution_mean=mask_distribution_mean)

    anisotropy_bounds = {
      violation_probability: compute_anisotropy_bound(
        diag_deviation_ess_sup=mask_distribution_ess_sup,
        subspace_coherence=subspace_coherence,
        subspace_dimension=subspace_basis.shape[1],
        diag_std_dev=mask_distribution_std_dev,
        bound_violation_probability=violation_probability
      )
      for violation_probability in anisotropy_bound_violation_probabilities
    }
  else:
    anisotropy_operator_norms = None
    anisotropy_bounds = None

  return SubspaceCoherenceResult(
    subspace_coherence=subspace_coherence,
    explained_variance_ratio=explained_variance_ratio,
    subspace_dimension=subspace_basis.shape[1],
    full_dimension=data.shape[1],
    anisotropy_operator_norms=anisotropy_operator_norms.tolist() if anisotropy_operator_norms is not None else None,
    mean_anisotropy_operator_norm=anisotropy_operator_norms.mean().item() if anisotropy_operator_norms is not None else None,
    anisotropy_bounds=anisotropy_bounds
  )

def compute_representation_subspace_coherences(
    activation_recording_points: List[ActivationRecordingPoint],
    model: torch.nn.Module,
    data_loader,
    target_explained_variance_ratio=0.9,
    device="cuda:0"
) -> Dict[ActivationRecordingPoint, SubspaceCoherenceResult]:
  """Collect activations per layer from `model` on data from `data_loader` and estimate their subspace coherences"""

  [recorded_activations,] = src.utils.record_activations.record_activations(
    activation_recording_points, models=[model], data_loader=data_loader, device=device
  )
  # Reshape to (n, d) format to be sure
  recorded_activations = {key: [val.reshape(val.shape[0], -1) for val in vals] for key, vals in recorded_activations.items()}

  named_modules = dict(model.named_modules())

  subspace_coherence_results_by_recording_point: Dict[ActivationRecordingPoint, SubspaceCoherenceResult] = {
    recording_point: compute_subspace_coherence_and_anisotropy(
      data=torch.cat(activations),
      target_explained_variance_ratio=target_explained_variance_ratio,
      layer=(layer := named_modules[recording_point[0]] if recording_point[1] == RecordInput else None), # type: ignore
      mask_distribution_mean=(
        layer.mask.mean().item()  # type: ignore
        if isinstance(layer, SparseLinear)
        else 1.
      ),
      mask_distribution_ess_sup=(
        layer.mask.mean().item()  # type: ignore
        if isinstance(layer, SparseLinear)
        else 0
      ),
      mask_distribution_std_dev=(
        ((p:=layer.mask.mean().item())*(1-p))**0.5  # type: ignore
        if isinstance(layer, SparseLinear)
        else 0
      )
    )
    for recording_point, activations in recorded_activations.items()
  }

  return subspace_coherence_results_by_recording_point

# Obtain a rotation matrix to rotate a dataset to approximately achieve a desired subspace coherence after PCA dimensionality reduction to subspace_dimension dimensions
def get_targeted_coherence_rotation(
  full_dimension: int,
  subspace_dimension: int,
  target_subspace_coherence: float,
  base_data: torch.Tensor | None = None,
  iters: int = 5000
) -> torch.Tensor:
  if target_subspace_coherence < subspace_dimension / full_dimension - 1e-5:
    raise ValueError(f"Impossible target: target_subspace_coherence={target_subspace_coherence} subspace_dimension/full_dimension={subspace_dimension/full_dimension:.4f}")

  dtype = base_data.dtype if base_data is not None else torch.float32
  device = base_data.device if base_data is not None else 'cpu'
  
  V = torch.eye(full_dimension, dtype=dtype, device=device)
  
  for _ in range(iters):
      norms = (V[:, :subspace_dimension]**2).sum(dim=1)
      if norms.max() <= target_subspace_coherence + 1e-5 or norms.max() - norms.min() < 1e-7: break
          
      i, j = norms.argmax().item(), norms.argmin().item()
      x, y, z = norms[i], norms[j], torch.dot(V[i, :subspace_dimension], V[j, :subspace_dimension])
      
      val = torch.clamp((max(target_subspace_coherence, (x+y)/2) - (x+y)/2) / torch.hypot((x-y)/2, -z), -1, 1)
      theta = 0.5 * (torch.atan2(-z, (x-y)/2) + torch.acos(val))
      
      c, s = torch.cos(theta).item(), torch.sin(theta).item()
      V[[i, j], :] = torch.tensor([[c, -s], [s, c]], dtype=V.dtype, device=V.device) @ V[[i, j], :]

  if base_data is None:
    return V
  return V @ torch.linalg.svd(base_data - base_data.mean(0), full_matrices=False)[2]

def compute_anisotropy_bound(
  diag_deviation_ess_sup: float, # b_d
  subspace_coherence: float, # nu(U)
  subspace_dimension: float, # k
  bound_violation_probability: float, # delta
  diag_std_dev: float, # sigma_d
) -> float:
  log_term = math.log(2*subspace_dimension/bound_violation_probability)
  return 2/3*diag_deviation_ess_sup*subspace_coherence*log_term + diag_std_dev*(subspace_coherence**0.5)*((2*log_term)**0.5)

class SubspaceCoherenceRotationTransform:
  def __init__(
      self,
      dataset,
      target_explained_variance_ratio: float,
      target_subspace_coherence: float,
      svd_device="cuda:0",
      transform_device="cpu"
    ):
    data = torch.stack([dataset[i][0].reshape(-1).to(svd_device) for i in range(len(dataset))])
    subspace_basis, _ = estimate_subspace_basis_at_explained_variance(data, target_explained_variance_ratio)
    transform_matrix = get_targeted_coherence_rotation(
      full_dimension=data.shape[1],
      subspace_dimension=subspace_basis.shape[1],
      target_subspace_coherence=target_subspace_coherence,
      base_data=data
    )
    self.transform_matrix = transform_matrix.to(transform_device)

  def __call__(self, input: torch.Tensor):
    assert torch.is_tensor(input)
    return (input.reshape(-1) @ self.transform_matrix.T).reshape(input.shape)
