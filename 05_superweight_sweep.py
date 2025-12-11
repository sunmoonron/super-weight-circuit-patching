import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from config import MODEL_NAME, REVISION, DTYPE, BASE_PATH, SUPERWEIGHTS, TEST_PROMPTS

def build_variants(base_state, key, row, col, w_star, k_spread=8):
    variants = {}

    # base
    st = copy.deepcopy(base_state)
    variants["base"] = st

    # zero
    st = copy.deepcopy(base_state)
    st[key][row, col] = 0.0
    variants["zero"] = st

    # half
    st = copy.deepcopy(base_state)
    st[key][row, col] = 0.5 * w_star
    variants["half"] = st

    # double
    st = copy.deepcopy(base_state)
    st[key][row, col] = 2.0 * w_star
    variants["double"] = st

    # sign flip
    st = copy.deepcopy(base_state)
    st[key][row, col] = -w_star
    variants["flip"] = st

    # spread-k: wipe that entry, spread onto neighbors
    st = copy.deepcopy(base_state)
    d_in = st[key].size(1)
    st[key][row, col] = 0.0
    for j_off in range(k_spread):
        j = (col + j_off) % d_in
        st[key][row, j] += w_star / k_spread
    variants[f"spread{ k_spread }"] = st

    return variants

def load_model_from_state(state):
    dtype = getattr(torch, DTYPE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=REVISION,
        torch_dtype=dtype,
        device_map=None,
    )
    model.load_state_dict(state)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tok

def generate(model, tok, prompt, max_new_tokens=20):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    base_state = load_file(BASE_PATH)
    sw = SUPERWEIGHTS[0]
    layer, row, col = sw["layer"], sw["row"], sw["col"]
    key = f"model.layers.{layer}.mlp.down_proj.weight"
    w_star = base_state[key][row, col].item()

    variants = build_variants(base_state, key, row, col, w_star, k_spread=8)

    # evaluate all variants on all prompts
    for prompt in TEST_PROMPTS:
        print("=" * 120)
        print("PROMPT:", repr(prompt))
        base_outs = {}
        for name, st in variants.items():
            model, tok = load_model_from_state(st)
            out = generate(model, tok, prompt)
            base_outs[name] = out
        # print in a consistent order
        for name in sorted(base_outs.keys()):
            print(f"{name.upper():8}:", base_outs[name])

if __name__ == "__main__":
    main()
