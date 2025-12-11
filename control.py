import torch
from safetensors.torch import load_file, save_file
from config import BASE_PATH, BROKEN_PATH, PATCH_PATH, SUPERWEIGHTS

base = load_file(BASE_PATH)
broken = load_file(BROKEN_PATH)

sw = SUPERWEIGHTS[0]
layer_idx, row, col = sw["layer"], sw["row"], sw["col"]
key = f"model.layers.{layer_idx}.mlp.down_proj.weight"

patch_state = {k: torch.zeros_like(v) for k, v in base.items()}
patch_state[key][row, col] = base[key][row, col] - broken[key][row, col]  # basically original_value

save_file(patch_state, PATCH_PATH)
