import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from datasets import load_dataset
from config import MODEL_NAME, REVISION, DTYPE, BASE_PATH, BROKEN_PATH

def load_model_from_state(path):
    dtype = getattr(torch, DTYPE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=REVISION,
        torch_dtype=dtype,
        device_map=None,
    )
    st = load_file(path)
    model.load_state_dict(st)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tok

def main():
    base_model, tok = load_model_from_state(BASE_PATH)
    zero_model, _ = load_model_from_state(BROKEN_PATH)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:1%]")
    scored_positions = []

    for ex in ds:
        text = ex["text"]
        if not text.strip():
            continue
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = enc["input_ids"]

        with torch.no_grad():
            base_out = base_model(input_ids=input_ids)
            zero_out = zero_model(input_ids=input_ids)

        # logits [1, T, V]
        b_logits = base_out.logits[:, :-1, :]
        z_logits = zero_out.logits[:, :-1, :]
        b_probs = F.softmax(b_logits, dim=-1)
        z_log_probs = F.log_softmax(z_logits, dim=-1)

        # KL per position
        kl = F.kl_div(z_log_probs, b_probs, reduction="none")  # [1, T-1, V]
        kl_pos = kl.sum(dim=-1).squeeze(0)  # [T-1]

        # store top few positions in this example
        for pos in range(kl_pos.size(0)):
            scored_positions.append(
                (kl_pos[pos].item(), text, pos)
            )

    # take global top 30
    scored_positions.sort(reverse=True, key=lambda x: x[0])
    top = scored_positions[:30]

    for score, text, pos in top:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
        ids = enc["input_ids"][0]
        token_str = tok.decode(ids[pos:pos+1])
        ctx = tok.decode(ids[max(0, pos-10):pos+10])
        print("="*80)
        print(f"Score: {score:.4f} Token: {repr(token_str)}")
        print("Context:", repr(ctx))

if __name__ == "__main__":
    main()
