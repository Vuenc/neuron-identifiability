import copy
from omegaconf import OmegaConf

def apply_n_mul_to_mask_params(mask_params, n_mul: float, is_top_level=True):
    """Recursively apply n_mul multiplier to all num_fixed values in mask_params.

    Args:
        mask_params: Dictionary containing mask parameters (may be nested, may be OmegaConf DictConfig)
        n_mul: Multiplier to apply to num_fixed values
        is_top_level: Whether this is the top-level call (needed to convert OmegaConf only once)

    Returns:
        New dictionary with num_fixed values multiplied by n_mul
    """
    # Convert OmegaConf DictConfig to regular dict if needed (only at top level)
    if is_top_level:
        try:
            # Try to convert if it's an OmegaConf object
            result = OmegaConf.to_container(mask_params, resolve=True)
        except (AttributeError, TypeError, ValueError):
            # If it's already a regular dict or not OmegaConf, just deepcopy
            result = copy.deepcopy(mask_params)
    else:
        # For nested dicts, they should already be regular dicts after top-level conversion
        result = copy.deepcopy(mask_params)

    if isinstance(result, dict):
        for key, value in list(result.items()):  # Use list() to avoid mutation during iteration
            if key == 'num_fixed' and isinstance(value, (int, float)):
                result[key] = int(value * n_mul)
            elif isinstance(value, dict):
                result[key] = apply_n_mul_to_mask_params(value, n_mul, is_top_level=False)

    return result
