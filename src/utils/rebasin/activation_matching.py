import collections
from typing import DefaultDict, Dict, List, Tuple

import torch
from scipy.optimize import linear_sum_assignment

from .common import PermutationSpec, MatchingResult, HookMode, LayerName, ActivationCorrelationMode

def activation_matching(
    permutation_spec: PermutationSpec,
    model_a,
    model_b,
    data_loader,
    normalize_similarities: bool=True,
    correlation_modes: List[ActivationCorrelationMode] = [ActivationCorrelationMode.DotProduct, ActivationCorrelationMode.PearsonCorrelation],
    device="cuda:0"
) -> Dict[Tuple[str, ActivationCorrelationMode], Dict[LayerName, MatchingResult]]:
  """Find a permutation of `params_b` to make them match `params_a`."""

  recorded_activations_by_mode_a: Dict[str, DefaultDict[LayerName, List[torch.Tensor]]] = {}
  recorded_activations_by_mode_b: Dict[str, DefaultDict[LayerName, List[torch.Tensor]]] = {}

  # Register hooks to save activations from layers that should be permuted
  for (model, recorded_activations_by_mode) in (model_a, recorded_activations_by_mode_a), (model_b, recorded_activations_by_mode_b):
    for activation_matching_mode_name, perm_to_hook_description in permutation_spec.activation_matching_modes.items():
      recorded_activations_by_mode[activation_matching_mode_name] = collections.defaultdict(lambda: [])
      recorded_activations_current_mode = recorded_activations_by_mode[activation_matching_mode_name]
      named_modules = dict(model.named_modules())
      for (layer_name, hook_mode) in perm_to_hook_description.values():
        if hook_mode == HookMode.RecordInput:
          def record_input_hook(module, input, output, layer_name=layer_name, recorded_activations_current_mode=recorded_activations_current_mode):
            recorded_activations_current_mode[layer_name].append(input[0])
          forward_hook = record_input_hook
        else:
          def record_output_hook(module, input, output, layer_name=layer_name, recorded_activations_current_mode=recorded_activations_current_mode):
            recorded_activations_current_mode[layer_name].append(output)
          forward_hook = record_output_hook
        named_modules[layer_name].register_forward_hook(forward_hook)

  # Forward the dataset through the models
  with torch.no_grad():
    # Only iterate the data loader once, so the models see the data in the same order
    for input, _ in data_loader:
      for model in [model_a, model_b]:
        model.forward(input.to(device))

  # Compute efficient permutations
  matching_results_by_mode: DefaultDict[Tuple[str, ActivationCorrelationMode], Dict[LayerName, MatchingResult]] = collections.defaultdict(lambda: {})
  for activation_matching_mode_name, perm_to_hook_description in permutation_spec.activation_matching_modes.items():
    for layer_name, _ in perm_to_hook_description.values():
      activations_a = torch.cat(recorded_activations_by_mode_a[activation_matching_mode_name][layer_name])
      activations_b = torch.cat(recorded_activations_by_mode_b[activation_matching_mode_name][layer_name])

      for correlation_mode in correlation_modes:
        if correlation_mode == ActivationCorrelationMode.DotProduct:
          # Using Z1 * Z2.T as similarity measure (see Git Rebasin discussion)
          activation_similarities = (activations_a.T @ activations_b)/(activations_a.shape[0] if normalize_similarities else 1)
        elif correlation_mode == ActivationCorrelationMode.PearsonCorrelation:
          centered_activations_a, centered_activations_b = [act - act.mean(dim=0, keepdim=True) for act in [activations_a, activations_b]]
          activation_similarities = (centered_activations_a / centered_activations_a.norm(dim=0)).T @ (centered_activations_b / centered_activations_b.norm(dim=0))
        else:
          raise ValueError(f"Unsupported activation correlation mode: {correlation_mode}")
        _, permutation = linear_sum_assignment(activation_similarities.detach().cpu().numpy(), maximize=True)
        matching_results_by_mode[(activation_matching_mode_name, correlation_mode)][layer_name] = MatchingResult(torch.as_tensor(permutation, device=activation_similarities.device), activation_similarities)

  return dict(matching_results_by_mode)
