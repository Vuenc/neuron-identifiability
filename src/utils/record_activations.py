import torch
from collections import defaultdict
import enum
from typing import DefaultDict, Dict, List, Tuple

type LayerName = str

class HookMode(enum.Enum):
  RecordInput = 0
  RecordOutput = 1

RecordInput, RecordOutput = HookMode.RecordInput, HookMode.RecordOutput

type ActivationRecordingPoint = Tuple[LayerName, HookMode]
type RecordedActivations = Dict[ActivationRecordingPoint, List[torch.Tensor]]

def record_activations(
    activation_recording_points: List[ActivationRecordingPoint],
    models: List[torch.nn.Module],
    data_loader,
    device="cuda:0",
    activations_target_device=None
) -> List[RecordedActivations]:
    recorded_activations_by_model: List[DefaultDict[ActivationRecordingPoint, List[torch.Tensor]]] = [defaultdict(lambda: []) for _ in models]
    if activations_target_device is None:
        activations_target_device = device

    registered_hooks = []
    # Register hooks to save activations from layers that should be permuted
    #   for (model, recorded_activations_by_mode) in (model_a, recorded_activations_by_mode_a), (model_b, recorded_activations_by_mode_b):
    for layer_name, hook_mode in activation_recording_points:
        for model, recorded_activations in zip(models, recorded_activations_by_model):
            named_modules = dict(model.named_modules())
            if hook_mode == RecordInput:
                def record_input_hook(module, input, output, layer_name=layer_name, hook_mode=hook_mode, recorded_activations=recorded_activations):
                    recorded_activations[(layer_name, hook_mode)].append(input[0].detach().to(activations_target_device))
                forward_hook = record_input_hook
            elif hook_mode == RecordOutput:
                def record_output_hook(module, input, output, layer_name=layer_name, hook_mode=hook_mode, recorded_activations=recorded_activations):
                    recorded_activations[(layer_name, hook_mode)].append(output.detach().to(activations_target_device))
                forward_hook = record_output_hook
            else:
                raise ValueError(f"Unsupported value for hook_mode: {hook_mode}")
            handle = named_modules[layer_name].register_forward_hook(forward_hook)
            registered_hooks.append(handle)

    # Forward the dataset through the models
    with torch.no_grad():
        # Only iterate the data loader once, so the models see the data in the same order
        for input, _ in data_loader:
            for model in models:
                model.forward(input.to(device))
            # import psutil
            # print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.1f} / {psutil.virtual_memory().total / 1e9:.1f} GB")
    for handle in registered_hooks:
        handle.remove()
    return [dict(recorded_activations) for recorded_activations in recorded_activations_by_model]
