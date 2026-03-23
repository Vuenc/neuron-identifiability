import collections
from typing import DefaultDict, Dict, List, Tuple

import torch
from scipy.optimize import linear_sum_assignment

from .common import PermutationSpec, MatchingResult, PermutationName, ActivationCorrelationMode
from src.utils.record_activations import record_activations, ActivationRecordingPoint

def activation_matching(
    permutation_spec: PermutationSpec,
    model_a,
    model_b,
    data_loader,
    normalize_similarities: bool=True,
    correlation_modes: List[ActivationCorrelationMode] = [ActivationCorrelationMode.DotProduct, ActivationCorrelationMode.PearsonCorrelation],
    device="cuda:0",
    verbose: bool=False,
) -> Dict[Tuple[str, ActivationCorrelationMode], Dict[PermutationName, MatchingResult]]:
  """Find a permutation of `params_b` to make them match `params_a`."""
  # Compile all points in the architecture where activations need to be recorded (avoid duplicates)
  activation_recording_points = list(set(
    recording_point
    for perm_to_recording_points in permutation_spec.activation_matching_modes.values()
    for recording_points in perm_to_recording_points.values()
    for recording_point in recording_points
  ))

  # Record activations of both models
  if verbose: print("Recording activations...")
  recorded_activations_a, recorded_activations_b = (
    record_activations(
      activation_recording_points=activation_recording_points,
      models=[model_a, model_b],
      data_loader=data_loader,
      device=device,
      activations_target_device="cpu"
  ))

  # Compute efficient permutations
  matching_results_by_mode: DefaultDict[Tuple[str, ActivationCorrelationMode], Dict[PermutationName, MatchingResult]] = collections.defaultdict(lambda: {})
  for activation_matching_mode_name, perm_to_hook_description in permutation_spec.activation_matching_modes.items():
    for permutation_name, recording_points in perm_to_hook_description.items():

      if verbose: print(f"Matching in {activation_matching_mode_name}: {recording_points}")
      activations_a = torch.cat([activations for recording_point in recording_points for activations in recorded_activations_a[recording_point]]).to(device)
      activations_b = torch.cat([activations for recording_point in recording_points for activations in recorded_activations_b[recording_point]]).to(device)

      if permutation_spec.reshape_activation_tensor is not None:
        # Reshape activation tensors into (dataset size) x (hidden dimensions) shape:
        # necessary e.g. for ResNets where at this stage they are (dataset size) x (num channels) x width x height
        # It's up to the PermutationSpec how exactly this reshaping should be implemented
        activations_a = permutation_spec.reshape_activation_tensor(activations_a)
        activations_b = permutation_spec.reshape_activation_tensor(activations_b)

      with torch.no_grad():
        for correlation_mode in correlation_modes:
          if correlation_mode is ActivationCorrelationMode.DotProduct:
            # Using Z1 * Z2.T as similarity measure (see Git Rebasin discussion)
            activation_similarities = (activations_a.T @ activations_b)/(activations_a.shape[0] if normalize_similarities else 1)
          elif correlation_mode is ActivationCorrelationMode.PearsonCorrelation or correlation_mode is ActivationCorrelationMode.PearsonCorrelationWithZeroForConstant:
            centered_activations_a, centered_activations_b = [act - act.mean(dim=0, keepdim=True) for act in [activations_a, activations_b]]
            activation_similarities = (centered_activations_a / centered_activations_a.norm(dim=0)).T @ (centered_activations_b / centered_activations_b.norm(dim=0))
            if correlation_mode is ActivationCorrelationMode.PearsonCorrelationWithZeroForConstant:
              activation_similarities.nan_to_num_(nan=0.)
            elif activation_similarities.isnan().any():
              raise ValueError(f"Constant hidden dimensions present at {recording_points}, cannot compute Pearson correlation. Use PearsonCorrelationWithZeroForConstant to replace these correlations with 0.")
          elif correlation_mode is ActivationCorrelationMode.PearsonUncorrelatedness or correlation_mode is ActivationCorrelationMode.PearsonUncorrelatednessWithOneForConstant:
            centered_activations_a, centered_activations_b = [act - act.mean(dim=0, keepdim=True) for act in [activations_a, activations_b]]
            activation_similarities = 1 - torch.abs((centered_activations_a / centered_activations_a.norm(dim=0)).T @ (centered_activations_b / centered_activations_b.norm(dim=0)))
            if correlation_mode is ActivationCorrelationMode.PearsonUncorrelatednessWithOneForConstant:
              activation_similarities.nan_to_num_(nan=1.)
            elif activation_similarities.isnan().any():
              raise ValueError(f"Constant hidden dimensions present at {recording_points}, cannot compute Pearson correlation. Use PearsonUncorrelatednessWithOneForConstant to replace these correlations with 1.")
          else:
            raise ValueError(f"Unsupported activation correlation mode: {correlation_mode}")
          activation_similarities = activation_similarities.detach()
          _, permutation = linear_sum_assignment(activation_similarities.cpu().numpy(), maximize=True)
          matching_results_by_mode[(activation_matching_mode_name, correlation_mode)][permutation_name] = MatchingResult(torch.as_tensor(permutation, device=activation_similarities.device), activation_similarities=activation_similarities)

  return dict(matching_results_by_mode)
