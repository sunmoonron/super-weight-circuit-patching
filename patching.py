import torch # Importing the main PyTorch library
import torch.nn as nn # Importing the neural network module from PyTorch
import torch.nn.functional as F # Importing functional interface for neural networks
from safetensors.torch import save_file # Importing save_file function from safetensors library

from config import PATCH_PATH # Importing configuration variables

# Compute y_base = x @ W^T
# y_patched[..., row_idx] += x @ delta_row^T # To flow gradients ONLY into delta_row
# To do above computations, we ensure storage of base weights as frozen buffer
# Also train only a single learnable row (i.e delta_row)

class DownProjPatchRow(nn.Module):
    """ 
    Wraps the original down_proj in one layer and adds a trainable delta to a single row (row_idx) of
    its weight matrix. 

    W' = W + E_row * delta_row
    """
    def __init__(self, base_linear: nn.Linear, row_idx: int):
        """
        base_linear: The original down_proj linear layer from the model
        row_idx: The index of the row to be patched
        """
        super().__init__()

        assert isinstance(base_linear, nn.Linear), "down_proj must be nn.Linear"
        assert base_linear.bias is None or base_linear.bias is False, "OLMo down_proj shouldn't have bias"

        # Store original weight as a non-trainable buffer
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.row_idx = row_idx

        self.register_buffer("base_weight", base_linear.weight.data.clone())

        # Trainable patch only for this row
        self.delta_row = nn.Parameter(torch.zeros(self.in_features))
        self.delta_row.requires_grad_(True)

    def forward(self, x):
        """
        x: [batch, seq, in_features]
        returns: [batch, seq, out_features]
        """
        
        # Base output
        y_base = F.linear(x, self.base_weight) # [B, T, out_features]

        # Patch contribution: x @ delta_row^T -> [B, T]
        patch_out = torch.matmul(x, self.delta_row.unsqueeze(-1)).squeeze(-1)

        # Add patch only to row_idx
        y = y_base.clone()
        y[..., self.row_idx] += patch_out
        return y
    
def apply_patch_to_olmo(model, layer_idx: int, row_idx: int):
    """
    Monkey-patch model.model.layers[layer_idx].mlp.down_proj with DownProjPatchRow.
    Args:
        model: OlmoForCausalLM instance (AutoModelForCausalLM from HF)
        layer_idx: which decoder layer (int)
        row_idx: which row in down_proj.weight to patch (int)
    Returns:
        (patched_mode, patch_module)
    """
    # OLMoForCausalLM has .model: OlmoModel
    decoded_layer = model.model.layers[layer_idx]
    mlp = decoded_layer.mlp
    orig_down = mlp.down_proj

    patched = DownProjPatchRow(orig_down, row_idx)

    mlp.down_proj = patched
    return model, patched
    
def build_patch_state_dict_like(
    base_state: dict, 
    layer_idx: int, 
    row_idx: int, 
    delta_row: torch.Tensor,
    ) -> dict:
    """
    Create a full-state patch dict (all zeroes except the patched row), so we can
    save it as a separate safetensors file.
    """
    patch_state = {k: torch.zeros_like(v) for k, v in base_state.items()}

    key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
    if key not in patch_state:
        raise KeyError(f"Key {key} not found in state dict")
    
    W = patch_state[key]
    assert row_idx < W.size(0), "row_idx out of range for down_proj.weight"
    assert delta_row.numel() == W.size(1), "delta_row length must equal in_features of down_proj"
    
    W[row_idx, :] = delta_row
    return patch_state

def save_patch_safetensors(
    base_state_dict: dict, 
    layer_idx: int, 
    row_idx: int, 
    delta_row: torch.Tensor):
    """
    Serialize the learned delta into PATCH_PATH as safetensors file
    """
    patch_state = build_patch_state_dict_like(
        base_state=base_state_dict, 
        layer_idx=layer_idx, 
        row_idx=row_idx, 
        delta_row=delta_row,
        )
    save_file(patch_state, PATCH_PATH)
    print(f"Saved patch to {PATCH_PATH}")

__all__ = ["DownProjPatchRow","apply_patch_to_olmo","save_patch_safetensors"]