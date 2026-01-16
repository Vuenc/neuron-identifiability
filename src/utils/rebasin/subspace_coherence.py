import collections
from typing import DefaultDict, Dict, List, Tuple

import torch

from .common import PermutationSpec, MatchingResult, HookMode, LayerName, ActivationCorrelationMode

SubspaceCoherenceResult = collections.namedtuple("SubspaceCoherenceResult", ["subspace_coherence", "explained_variance_ratio", "subspace_dimension", "full_dimension"])

def subspace_coherence_at_explained_variance(data, target_explained_variance_ratio) -> SubspaceCoherenceResult:
  data = data - data.mean(dim=0)
  _, S, UT = torch.linalg.svd(data, full_matrices=False)
  U = UT.T
  explained_variance_ratio_cum = torch.cumsum((S**2) / (S**2).sum(), 0)
  k: int = torch.searchsorted(
      explained_variance_ratio_cum,
      target_explained_variance_ratio
  ).item() # type: ignore
  subspace_coherence = (U[:, :k]**2).sum(dim=1).max().item()
  return SubspaceCoherenceResult(
    subspace_coherence=subspace_coherence,
    explained_variance_ratio=explained_variance_ratio_cum[k].item(),
    subspace_dimension=k,
    full_dimension=data.shape[1],
  )

def compute_representation_subspace_coherences(
    permutation_spec: PermutationSpec,
    model,
    data_loader,
    target_explained_variance_ratio=0.9,
    device="cuda:0"
) -> Dict[str, Dict[LayerName, SubspaceCoherenceResult]]:
  """Find a permutation of `params_b` to make them match `params_a`."""

  recorded_activations_by_mode: Dict[str, DefaultDict[LayerName, List[torch.Tensor]]] = {"input_data": collections.defaultdict(lambda: [])}

  # Register hooks to save activations from layers that should be permuted
  for activation_matching_mode_name, perm_to_hook_description in permutation_spec.activation_matching_modes.items():
    recorded_activations_by_mode[activation_matching_mode_name] = collections.defaultdict(lambda: [])
    recorded_activations_current_mode = recorded_activations_by_mode[activation_matching_mode_name]
    named_modules = dict(model.named_modules())
    for (layer_name, hook_mode) in perm_to_hook_description.values():
      if layer_name not in named_modules:
        print(f"Warning: Layer {layer_name} not found in model. Not recording activations for this layer.")
        continue
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
      input = input.to(device)
      model.forward(input.to(device))
      recorded_activations_by_mode["input_data"]["input"].append(input.view(input.shape[0], -1))

  # Compute efficient permutations
  subspace_coherence_result_by_mode: DefaultDict[str, Dict[LayerName, SubspaceCoherenceResult]] = collections.defaultdict(lambda: {})
  for activation_matching_mode_name, collected_activations in recorded_activations_by_mode.items():
    for layer_name, collected_activations_layer in collected_activations.items():
      activations = torch.cat(collected_activations_layer)
      subspace_coherence_result = subspace_coherence_at_explained_variance(activations, target_explained_variance_ratio)
      subspace_coherence_result_by_mode[activation_matching_mode_name][layer_name] = subspace_coherence_result

  return dict(subspace_coherence_result_by_mode)
