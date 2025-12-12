# Run this on CPU (requires ~16GB RAM)
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

from config import (
    MODEL_NAME,
    REVISION,
    DTYPE,
    SUPERWEIGHTS,
    BASE_PATH,
    BROKEN_PATH,
)

def load_olmo():
    dtype = getattr(torch, DTYPE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=REVISION,
        torch_dtype=dtype,
        device_map=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

def main():
    model, tokenizer = load_olmo()
    state = model.state_dict()

    # Save base
    save_file(state, BASE_PATH)
    print(f"Saved base checkpoint to {BASE_PATH}")

    # Utilize the first superweight 
    sw = SUPERWEIGHTS[0]
    layer_idx = sw["layer"]
    row = sw["row"]
    col = sw["col"]

    key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
    W = state[key]
    print("down_proj shape:", W.shape)
    assert row < W.size(0) and col < W.size(0), "Superweight indices out of range"

    original_value = W[row, col].item()
    print(f"Original superweight at layer {layer_idx}, row {row}, col {col} = {original_value}")

    broken_state = copy.deepcopy(state)
    with torch.no_grad():
        broken_state[key][row, col] = 0.0
    save_file(broken_state, BROKEN_PATH)
    print(f"Saved broken checkpoint (superweight zeroed) to {BROKEN_PATH}")

if __name__ == "__main__":
    main()