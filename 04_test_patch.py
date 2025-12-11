import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file

from config import (
    MODEL_NAME,
    REVISION,
    DTYPE,
    BASE_PATH,
    BROKEN_PATH,
    PATCH_PATH,
    PATCHED_MERGED_PATH,
    TEST_PROMPTS,
)

def load_model_from_state(path):
    dtype = getattr(torch, DTYPE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=REVISION,
        torch_dtype=dtype,
        device_map=None,
    )
    state = load_file(path)
    model.load_state_dict(state)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

def generate(model, tokenizer, prompt, max_new_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def main():
    base_state = load_file(BASE_PATH)
    broken_state = load_file(BROKEN_PATH)
    patch_state = load_file(PATCH_PATH)

    # Merge broken + patch
    merged_state = {}
    for k in broken_state.keys():
        merged_state[k] = broken_state[k] + patch_state[k]
    
    save_file(merged_state, PATCHED_MERGED_PATH)
    print(f"Saved merged patched checkpoint to {PATCHED_MERGED_PATH}")

    base_model, tok = load_model_from_state(BASE_PATH)
    broken_mode, _ = load_model_from_state(BROKEN_PATH)
    patched_model, _ = load_model_from_state(PATCHED_MERGED_PATH)

    for p in TEST_PROMPTS:
        print("=" * 80)
        print("PROMPT:", repr(p))
        base_out = generate(base_model, tok, p)
        broken_out = generate(broken_mode, tok, p)
        patched_out = generate(patched_model, tok, p)

        print("BASE.  :", base_out)
        print("BROKEN :", broken_out)
        print("PATCHED:", patched_out)

if __name__ == "__main__":
    main()