import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

from config import MODEL_NAME, REVISION, DTYPE, BASE_PATH, BROKEN_PATH, TEST_PROMPTS

def load_from_safetensors(path):
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
            do_sample=False, # greedy
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def main():
    base_model, tok = load_from_safetensors(BASE_PATH)
    broken_model, _ = load_from_safetensors(BROKEN_PATH)

    for p in TEST_PROMPTS:
        print("=" * 80)
        print("PROMPT:", repr(p))
        base_out = generate(base_model, tok, p)
        broken_out = generate(broken_model, tok, p)
        print("BASE  :", base_out)
        print("BROKEN:", broken_out)

if __name__ == "__main__":
    main()