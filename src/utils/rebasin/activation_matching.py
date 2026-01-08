from collections import defaultdict
import collections
from typing import Callable, Dict, NamedTuple

import torch
from scipy.optimize import linear_sum_assignment
import tqdm
import numpy as np


class PermutationSpec(NamedTuple):
  perm_to_axes: dict
  axes_to_perm: dict
  perm_to_activations_layer_names: dict[str, str]
  perm_to_input_layer_names: dict[str, str]
  reshape_activation_tensor: Callable[[torch.Tensor], torch.Tensor] | None=None

class MatchingResult(NamedTuple):
  optimal_permutation: torch.Tensor
  activation_similarities: torch.Tensor

def permutation_spec_from_axes_to_perm(axes_to_perm: dict, perm_to_activations_layer_names, perm_to_input_layer_names) -> PermutationSpec:
  perm_to_axes = defaultdict(list)
  for wk, axis_perms in axes_to_perm.items():
    for axis, perm in enumerate(axis_perms):
      if perm is not None:
        perm_to_axes[perm].append((wk, axis))
  return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm,
                         perm_to_activations_layer_names=perm_to_activations_layer_names, perm_to_input_layer_names=perm_to_input_layer_names)

def mlp_permutation_spec(num_hidden_layers: int, norm = False, bias = True) -> PermutationSpec:
  """We assume that one permutation cannot appear in two axes of the same weight array."""
  assert num_hidden_layers >= 1
  perm_to_activations_layer_names = {
    f"P_{i}": f"lins.{i}" for i in range(num_hidden_layers)
  }
  perm_to_input_layer_names = {
    f"P_{i}": f"lins.{i}" for i in range(1, num_hidden_layers+1)
  }
  if not norm:
      if bias:
          return permutation_spec_from_axes_to_perm({
              "lins.0.weight": ("P_0", None),
              **{f"lins.{i}.weight": ( f"P_{i}", f"P_{i-1}")
                 for i in range(1, num_hidden_layers)},
              **{f"lins.{i}.bias": (f"P_{i}", )
                 for i in range(num_hidden_layers)},
              f"lins.{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
              f"lins.{num_hidden_layers}.bias": (None, ),
          },
          perm_to_activations_layer_names=perm_to_activations_layer_names, perm_to_input_layer_names=perm_to_input_layer_names)
      else:
          return permutation_spec_from_axes_to_perm({
              "lins.0.weight": ("P_0", None),
              **{f"lins.{i}.weight": ( f"P_{i}", f"P_{i-1}")
                 for i in range(1, num_hidden_layers)},
              f"lins.{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
          },
          perm_to_activations_layer_names=perm_to_activations_layer_names, perm_to_input_layer_names=perm_to_input_layer_names)
  else:
    return permutation_spec_from_axes_to_perm({
      "lins.0.weight": ("P_0", None),
      **{f"lins.{i}.weight": ( f"P_{i}", f"P_{i-1}")
         for i in range(1, num_hidden_layers)},
      **{f"lins.{i}.bias": (f"P_{i}", )
         for i in range(num_hidden_layers)},
      **{f"norms.{i}.weight": (f"P_{i}", )
         for i in range(num_hidden_layers)},
      **{f"norms.{i}.bias": (f"P_{i}", )
         for i in range(num_hidden_layers)},
      f"lins.{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
      f"lins.{num_hidden_layers}.bias": (None, ),
    },
    perm_to_activations_layer_names=perm_to_activations_layer_names, perm_to_input_layer_names=perm_to_input_layer_names)
      

"""
def cnn_permutation_spec() -> PermutationSpec:
  conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None, )}
  dense = lambda name, p_in, p_out, bias=True: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )} if bias else  {f"{name}.weight": (p_out, p_in)}

  return permutation_spec_from_axes_to_perm({
     **conv("conv1", None, "P_bg0"),
     **conv("conv2", "P_bg0", "P_bg1"),
     **dense("fc1", "P_bg1", "P_bg2"),
     **dense("fc2", "P_bg2", None, False),
  })
"""



# def resnet20_permutation_spec() -> PermutationSpec:
#   conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None, )}
#   norm = lambda name, p: {f"{name}.weight": (p, ), f"{name}.bias": (p, )}
#   dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}

#   # This is for easy blocks that use a residual connection, without any change in the number of channels.
#   easyblock = lambda name, p: {
#       **conv(f"{name}.conv1", p, f"P_{name}_inner"),
#       **norm(f"{name}.ln1", f"P_{name}_inner"),
          
#       **conv(f"{name}.conv2", f"P_{name}_inner", p),
#       **norm(f"{name}.ln2", p),
      
#   }

#   # This is for blocks that use a residual connection, but change the number of channels via a Conv.
#   shortcutblock = lambda name, p_in, p_out: {
#       **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
#       **norm(f"{name}.ln1", f"P_{name}_inner"),
          
#       **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
#       **norm(f"{name}.ln2", p_out),
#       **conv(f"{name}.shortcut.0", p_in, p_out),
#       **norm(f"{name}.shortcut.1", p_out),

#   }
#   return permutation_spec_from_axes_to_perm({
#     **conv("conv1", None, "P_bg0"),
#     **norm("ln1", "P_bg0"),
#           #
#     **easyblock("layer1.0", "P_bg0"),
#     **easyblock("layer1.1", "P_bg0",),
#     **easyblock("layer1.2", "P_bg0"),
#           #
    
#     **shortcutblock("layer2.0", "P_bg0", "P_bg1"),
#     **easyblock("layer2.1", "P_bg1",),
#     **easyblock("layer2.2", "P_bg1"),
#     #**easyblock("layer2.3", "P_bg2"),
    
#     **shortcutblock("layer3.0", "P_bg1", "P_bg2"),
#     **easyblock("layer3.1", "P_bg2",),
#     **easyblock("layer3.2", "P_bg2"),
#     #
#     **dense("linear", "P_bg2", None),
#   })

# # should be easy to generalize it to any depth
# def resnet50_permutation_spec() -> PermutationSpec:
#   conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None, )}
#   norm = lambda name, p: {f"{name}.weight": (p, ), f"{name}.bias": (p, )}
#   dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}

#   # This is for easy blocks that use a residual connection, without any change in the number of channels.
#   easyblock = lambda name, p: {
#   **norm(f"{name}.bn1", p),
#   **conv(f"{name}.conv1", p, f"P_{name}_inner"),
#   **norm(f"{name}.bn2", f"P_{name}_inner"),
#   **conv(f"{name}.conv2", f"P_{name}_inner", p),
#   }

#   # This is for blocks that use a residual connection, but change the number of channels via a Conv.
#   shortcutblock = lambda name, p_in, p_out: {
#   **norm(f"{name}.bn1", p_in),
#   **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
#   **norm(f"{name}.bn2", f"P_{name}_inner"),
#   **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
#   **conv(f"{name}.shortcut.0", p_in, p_out),
#   **norm(f"{name}.shortcut.1", p_out),
#   }

#   return permutation_spec_from_axes_to_perm({
#     **conv("conv1", None, "P_bg0"),
#     #
#     **shortcutblock("layer1.0", "P_bg0", "P_bg1"),
#     **easyblock("layer1.1", "P_bg1",),
#     **easyblock("layer1.2", "P_bg1"),
#     **easyblock("layer1.3", "P_bg1"),
#     **easyblock("layer1.4", "P_bg1"),
#     **easyblock("layer1.5", "P_bg1"),
#     **easyblock("layer1.6", "P_bg1"),
#     **easyblock("layer1.7", "P_bg1"),

#     #**easyblock("layer1.3", "P_bg1"),

#     **shortcutblock("layer2.0", "P_bg1", "P_bg2"),
#     **easyblock("layer2.1", "P_bg2",),
#     **easyblock("layer2.2", "P_bg2"),
#     **easyblock("layer2.3", "P_bg2"),
#     **easyblock("layer2.4", "P_bg2"),
#     **easyblock("layer2.5", "P_bg2"),
#     **easyblock("layer2.6", "P_bg2"),
#     **easyblock("layer2.7", "P_bg2"),

#     **shortcutblock("layer3.0", "P_bg2", "P_bg3"),
#     **easyblock("layer3.1", "P_bg3",),
#     **easyblock("layer3.2", "P_bg3"),
#     **easyblock("layer3.3", "P_bg3"),
#     **easyblock("layer3.4", "P_bg3"),
#     **easyblock("layer3.5", "P_bg3"),
#     **easyblock("layer3.6", "P_bg3"),
#     **easyblock("layer3.7", "P_bg3"),

#     **norm("bn1", "P_bg3"),

#     **dense("linear", "P_bg3", None),

# })



# def vgg16_permutation_spec() -> PermutationSpec:
#   layers_with_conv = [3,7,10,14,17,20,24,27,30,34,37,40]
#   layers_with_conv_b4 = [0,3,7,10,14,17,20,24,27,30,34,37]
#   layers_with_bn = [4,8,11,15,18,21,25,28,31,35,38,41]
#   dense = lambda name, p_in, p_out, bias = True: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}
#   return permutation_spec_from_axes_to_perm({
#       # first features
#       "features.0.weight": ( "P_Conv_0",None, None, None),
#       "features.1.weight": ( "P_Conv_0", None),
#       "features.1.bias": ( "P_Conv_0", None),
#       "features.1.running_mean": ( "P_Conv_0", None),
#       "features.1.running_var": ( "P_Conv_0", None),
#       "features.1.num_batches_tracked": (),

#       **{f"features.{layers_with_conv[i]}.weight": ( f"P_Conv_{layers_with_conv[i]}", f"P_Conv_{layers_with_conv_b4[i]}", None, None, )
#         for i in range(len(layers_with_conv))},
#       **{f"features.{i}.bias": (f"P_Conv_{i}", )
#         for i in layers_with_conv + [0]},
#       # bn
#       **{f"features.{layers_with_bn[i]}.weight": ( f"P_Conv_{layers_with_conv[i]}", None)
#         for i in range(len(layers_with_bn))},
#       **{f"features.{layers_with_bn[i]}.bias": ( f"P_Conv_{layers_with_conv[i]}", None)
#         for i in range(len(layers_with_bn))},
#       **{f"features.{layers_with_bn[i]}.running_mean": ( f"P_Conv_{layers_with_conv[i]}", None)
#         for i in range(len(layers_with_bn))},
#       **{f"features.{layers_with_bn[i]}.running_var": ( f"P_Conv_{layers_with_conv[i]}", None)
#         for i in range(len(layers_with_bn))},
#       **{f"features.{layers_with_bn[i]}.num_batches_tracked": ()
#         for i in range(len(layers_with_bn))},

#       **dense("classifier", "P_Conv_40", "P_Dense_0", False),
# })

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
  """Get parameter `k` from `params`, with the permutations applied."""
  w = params[k]
  if k not in ps.axes_to_perm:
    print(f"Skipping {k}")
    return w
  for axis, p in enumerate(ps.axes_to_perm[k]):
    # Skip the axis we're trying to permute.
    if axis == except_axis:
      continue

    # None indicates that there is no permutation relevant to that axis.
    if p is not None:
        w = torch.index_select(w, axis, perm[p].int())

  return w

def apply_permutation(ps: PermutationSpec, perm, params):
  """Apply a `perm` to `params`."""
  return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}

def activation_matching(permutation_spec: PermutationSpec, model_a, model_b, data_loader, normalize_similarities: bool=True, device="cuda:0") -> Dict[str, MatchingResult]:
  """Find a permutation of `params_b` to make them match `params_a`."""

  # params_a, params_b = model_a.state_dict(), model_b.state_dict()
  # perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in permutation_spec.perm_to_axes.items()}

  input_layer_names = permutation_spec.perm_to_input_layer_names.values()
  activation_layer_names = permutation_spec.perm_to_activations_layer_names.values()
  
  layer_inputs_by_layer_name_a, layer_inputs_by_layer_name_b = collections.defaultdict(lambda: []), collections.defaultdict(lambda: [])
  activations_by_layer_name_a, activations_by_layer_name_b = collections.defaultdict(lambda: []), collections.defaultdict(lambda: [])
  for model, activations_by_layer_name, layer_inputs_by_layer_name in [
      (model_a, activations_by_layer_name_a, layer_inputs_by_layer_name_a),
      (model_b, activations_by_layer_name_b, layer_inputs_by_layer_name_b)
  ]:
    # Register hooks to save activations from layers that should be permuted
    named_modules = dict(model.named_modules())
    for layer_name in input_layer_names:
      def forward_hook(module, input, output, layer_name=layer_name):
        layer_inputs_by_layer_name[layer_name].append(input[0])
      named_modules[layer_name].register_forward_hook(forward_hook)
    for layer_name in activation_layer_names:
      def forward_hook(module, input, output, layer_name=layer_name):
        activations_by_layer_name[layer_name].append(output)
      named_modules[layer_name].register_forward_hook(forward_hook)

    # Forward the dataset through the model
    with torch.no_grad():
      for input, _ in data_loader:
        model.forward(input.to(device))

  # Compute efficient permutations
  matching_results_inputs = {}
  matching_results_outputs = {}
  for matching_results, activations_or_inputs_a, activations_or_inputs_b in [
    (matching_results_inputs, layer_inputs_by_layer_name_a, layer_inputs_by_layer_name_b),
    (matching_results_outputs, activations_by_layer_name_a, activations_by_layer_name_b),
  ]:
    for layer_name in activations_or_inputs_a.keys():
      activations_a = torch.cat(activations_or_inputs_a[layer_name])
      activations_b = torch.cat(activations_or_inputs_b[layer_name])

      # Using Z1 * Z2.T as similarity measure (see Git Rebasin discussion); we could also use Pearson correlation
      activation_similarities = (activations_a.T @ activations_b)/(activations_a.shape[0] if normalize_similarities else 1)
      _, permutation = linear_sum_assignment(activation_similarities.detach().cpu().numpy(), maximize=True)
      matching_results[layer_name] = MatchingResult(torch.as_tensor(permutation, device=activation_similarities.device), activation_similarities)

  return {
    **{f"{layer_name}.input": result for layer_name, result in matching_results_inputs.items()},
    **{f"{layer_name}.output": result for layer_name, result in matching_results_outputs.items()}
  }
