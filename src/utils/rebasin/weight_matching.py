from collections import defaultdict
from re import L
from typing import Any, Callable, Dict, List, NamedTuple, Tuple

import torch
from scipy.optimize import linear_sum_assignment
import tqdm

from src.utils.rebasin.common import PermutationName, PermutationSpec, get_permuted_param, mlp_permutation_spec

def weight_matching(
  permutation_spec: PermutationSpec,
  params_a,
  params_b,
  max_iter=100,
  init_perm=None,
  restarts=10,
  verbose=False,
) -> Dict[PermutationName, torch.Tensor]:
  """Find a permutation of `params_b` to make them match `params_a`."""
  perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in permutation_spec.perm_to_axes.items()}
  device = next(iter(params_a.values())).device

  original_all_layers_permutations: Dict[PermutationName, torch.Tensor] = {p: torch.arange(n, device=device) for p, n in perm_sizes.items()} if init_perm is None else init_perm
  permutation_names = list(original_all_layers_permutations.keys())

  perms_and_values: List[Tuple[Dict[PermutationName, torch.Tensor], int]] = []

  measure_l2_norm = lambda all_layers_permutations: sum(
      torch.linalg.norm((params_a[param_name] - get_permuted_param(permutation_spec, all_layers_permutations, param_name, params_b)).reshape(-1)).item()
      for param_name in permutation_spec.axes_to_perm
    )

  for restart in tqdm.tqdm(range(restarts)):
    from copy import deepcopy
    current_all_layers_permutations = deepcopy(original_all_layers_permutations)
    newL = 0
    for iteration in tqdm.tqdm(range(max_iter), leave=False):
      progress = False
      for permutation_index in torch.randperm(len(permutation_names)):
        permutation_name = permutation_names[permutation_index]
        n = perm_sizes[permutation_name]

        # A = get_permuted_weight(p, perm)
        A = torch.zeros((n, n), device=device)
        for param_name, axis in permutation_spec.perm_to_axes[permutation_name]:
          w_a = params_a[param_name]
          w_b = get_permuted_param(permutation_spec, current_all_layers_permutations, param_name, params_b, except_axis=axis)
          w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
          w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))
          A += w_a @ w_b.T

        ri, ci = linear_sum_assignment(A.detach().cpu().numpy(), maximize=True)
        assert (torch.tensor(ri) == torch.arange(len(ri))).all()
        ci = torch.tensor(ci, device=device)
        oldL = torch.einsum('ij,ij->i', A, torch.eye(n, device=device)[current_all_layers_permutations[permutation_name].long()]).sum()
        newL = torch.einsum('ij,ij->i', A,torch.eye(n, device=device)[ci, :]).sum()
        # print(f"{iteration}/{p}: {newL - oldL} ({oldL} -> {newL})")
        progress = progress or newL > oldL + 1e-12

        current_all_layers_permutations[permutation_name] = ci

      if not progress:
        break

    sum_of_l2_norms = measure_l2_norm(current_all_layers_permutations)
    perms_and_values.append((current_all_layers_permutations, sum_of_l2_norms))

  from collections import Counter
  found_solutions_counts = sorted(Counter([v for p, v in perms_and_values]).items(), reverse=False) # type: ignore
  if verbose:
    print("Found solutions with counts:", found_solutions_counts)
    print("Original l2 distance:", measure_l2_norm(original_all_layers_permutations))
    print("Best found l2 distance:", found_solutions_counts[0][0])
  return min(perms_and_values, key=lambda p_v: p_v[1])[0]
