import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file
from datasets import load_dataset

# ----------------------------
# CONFIG
# ----------------------------

MODEL_NAME = "allenai/OLMo-1B-0724-hf"
REVISION = None        # or a specific revision if you used one
DTYPE = "float32"      # keep as float32 on CPU

BASE_PATH = "base.safetensors"
BROKEN_PATH = "broken.safetensors"
PATCH_PATH = "patch.safetensors"
PATCHED_MERGED_PATH = "patched_merged.safetensors"

# superweight coordinates (layer, row, col) for OLMo-1B-0724-hf
SW_LAYER = 1
SW_ROW = 1764
SW_COL = 1710
DOWN_PROJ_KEY = f"model.layers.{SW_LAYER}.mlp.down_proj.weight"

# eval settings (tweak if too slow)
MAX_LEN = 128
BATCH_SIZE = 1
NUM_EVAL_BATCHES = 30  # number of wikitext batches for metrics

TEST_PROMPTS = [
    "Summer is hot. Winter is",
    "Paris is in France. Tokyo is in",
    "2, 4, 6, 8,",
    "The capital of Canada is",
    "He opened the quote \" and then wrote Hello. To close it he should type",
    "List of numers: [1, 2, 3",
    "In Python, to close a string started with ' you end with ",
    "The opposite of big is",
    "A cat is an animal. A rose is a",
    "Complete the analogy: day : night :: up :",
]


# ----------------------------
# UTILS
# ----------------------------

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


def make_batches(tokenizer, texts, batch_size=BATCH_SIZE, max_len=MAX_LEN):
    batch = []
    for t in texts:
        if not t.strip():
            continue
        enc = tokenizer(
            t,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        if enc["input_ids"].size(1) < 4:
            continue
        batch.append(enc)
        if len(batch) == batch_size:
            maxL = max(b["input_ids"].size(1) for b in batch)
            ids_list, attn_list = [], []
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            for b in batch:
                ids = b["input_ids"]
                L = ids.size(1)
                if L < maxL:
                    pad = torch.full((1, maxL - L), pad_id, dtype=torch.long)
                    ids = torch.cat([ids, pad], dim=1)
                ids_list.append(ids)
                attn_list.append((ids != pad_id).long())
            input_ids = torch.cat(ids_list, dim=0)
            attention_mask = torch.cat(attn_list, dim=0)
            yield {"input_ids": input_ids, "attention_mask": attention_mask}
            batch = []


def compute_metrics(base_model, cmp_model, tokenizer, texts, num_batches=NUM_EVAL_BATCHES):
    """
    Returns:
      - cmp_nll, cmp_ppl
      - avg KL(base || cmp)
    """
    total_nll = 0.0
    total_tokens = 0
    total_kl = 0.0
    total_kl_tokens = 0

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for batch_idx, batch in enumerate(make_batches(tokenizer, texts)):
        if batch_idx >= num_batches:
            break

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        with torch.no_grad():
            base_out = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            cmp_out = cmp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # shift for LM
        labels = input_ids[:, 1:].contiguous()
        mask = (labels != pad_id)

        b_logits = base_out.logits[:, :-1, :]
        c_logits = cmp_out.logits[:, :-1, :]

        # Cross-entropy NLL for cmp model vs true labels
        vocab_size = c_logits.size(-1)
        ce_loss = F.cross_entropy(
            c_logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=pad_id,
            reduction="sum",
        )
        num_valid = mask.sum().item()
        total_nll += ce_loss.item()
        total_tokens += num_valid

        # KL(base || cmp) per token
        t_probs = F.softmax(b_logits, dim=-1)
        s_log_probs = F.log_softmax(c_logits, dim=-1)
        kl_full = F.kl_div(s_log_probs, t_probs, reduction="none")  # [B,T,V]
        kl_pos = kl_full.sum(dim=-1)  # [B,T]
        total_kl += (kl_pos * mask).sum().item()
        total_kl_tokens += num_valid

    avg_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(avg_nll) if total_tokens > 0 else float("inf")
    avg_kl = total_kl / max(total_kl_tokens, 1)

    return avg_nll, ppl, avg_kl


def generate(model, tok, prompt, max_new_tokens=20):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tok.decode(out[0], skip_special_tokens=True)


# ----------------------------
# PATCH INSPECTION
# ----------------------------

def inspect_patch():
    base_state = load_file(BASE_PATH)
    broken_state = load_file(BROKEN_PATH)
    patch_state = load_file(PATCH_PATH)

    base_row = base_state[DOWN_PROJ_KEY][SW_ROW].clone()
    broken_row = broken_state[DOWN_PROJ_KEY][SW_ROW].clone()
    delta_row = patch_state[DOWN_PROJ_KEY][SW_ROW].clone()
    merged_row = broken_row + delta_row

    def norm(t): return t.norm().item()

    print("\n=== ROW NORM STATS ===")
    print("||base_row||   =", norm(base_row))
    print("||broken_row|| =", norm(broken_row))
    print("||delta_row||  =", norm(delta_row))
    print("||merged_row|| =", norm(merged_row))
    print("||base - merged|| =", norm(base_row - merged_row))

    # cosine similarities
    def cos(a, b):
        return F.cosine_similarity(a.view(1, -1), b.view(1, -1)).item()

    print("\n=== COSINE SIMILARITIES ===")
    print("cos(base, broken) =", cos(base_row, broken_row))
    print("cos(base, delta)  =", cos(base_row, delta_row))
    print("cos(base, merged) =", cos(base_row, merged_row))

    # top-k entries in delta_row
    K = 20
    topk = torch.topk(delta_row.abs(), k=K)
    idx = topk.indices
    vals = delta_row[idx]

    print(f"\n=== TOP {K} ENTRIES IN delta_row (index, delta, base_value) ===")
    for i, v in zip(idx.tolist(), vals.tolist()):
        print(f"  j={i:5d}  delta={v:+.5e}  base={base_row[i].item():+.5e}")


# ----------------------------
# SPARSITY EXPERIMENT (OPTIONAL)
# ----------------------------

def build_sparse_patch_state(base_patch_state, k):
    """Keep only top-k entries in delta_row."""
    patch_state = {kname: torch.zeros_like(v) for kname, v in base_patch_state.items()}
    delta_full = base_patch_state[DOWN_PROJ_KEY][SW_ROW].clone()
    if k <= 0:
        return patch_state

    topk = torch.topk(delta_full.abs(), k=k)
    idx = topk.indices
    vals = delta_full[idx]

    row_vec = patch_state[DOWN_PROJ_KEY][SW_ROW]
    row_vec[idx] = vals
    return patch_state


def sparsity_ablation():
    print("\n\n================ SPARSITY ABLATION (PROMPT-LEVEL) ================\n")
    broken_state = load_file(BROKEN_PATH)
    base_patch_state = load_file(PATCH_PATH)

    ks = [8, 16, 32, 64]

    # always include full patch as reference
    print("FULL PATCH:")
    full_merged = {k: broken_state[k] + base_patch_state[k] for k in broken_state.keys()}
    save_file(full_merged, "tmp_full_merged.safetensors")
    full_model, tok = load_model_from_state("tmp_full_merged.safetensors")
    for p in TEST_PROMPTS:
        print("-" * 80)
        print("PROMPT :", repr(p))
        print("OUTPUT :", generate(full_model, tok, p, max_new_tokens=20))
    print("\n")

    for k in ks:
        print(f"\n--- TOP-{k} SPARSE PATCH ---")
        sparse_patch_state = build_sparse_patch_state(base_patch_state, k)
        merged = {name: broken_state[name] + sparse_patch_state[name] for name in broken_state.keys()}
        tmp_path = f"tmp_sparse_top{k}.safetensors"
        save_file(merged, tmp_path)
        model, tok = load_model_from_state(tmp_path)
        for p in TEST_PROMPTS:
            print("-" * 80)
            print("PROMPT :", repr(p))
            print("OUTPUT :", generate(model, tok, p, max_new_tokens=20))


# ----------------------------
# MAIN
# ----------------------------

def main():
    print("Loading models...")
    base_model, tok = load_model_from_state(BASE_PATH)
    broken_model, _ = load_model_from_state(BROKEN_PATH)
    patched_model, _ = load_model_from_state(PATCHED_MERGED_PATH)

    print("\n=== METRICS ON WIKITEXT-2 (SMALL SLICE) ===")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:2%]")
    texts = [x["text"] for x in ds]

    # base NLL / ppl (no KL needed, KL=0 by definition)
    base_nll, base_ppl, _ = compute_metrics(base_model, base_model, tok, texts)
    print(f"BASE:    NLL={base_nll:.4f}  PPL={base_ppl:.2f}  KL(base||base)=0")

    # broken vs base
    broken_nll, broken_ppl, broken_kl = compute_metrics(base_model, broken_model, tok, texts)
    print(f"BROKEN:  NLL={broken_nll:.4f}  PPL={broken_ppl:.2f}  KL(base||broken)={broken_kl:.4f}")

    # patched vs base
    patched_nll, patched_ppl, patched_kl = compute_metrics(base_model, patched_model, tok, texts)
    print(f"PATCHED: NLL={patched_nll:.4f}  PPL={patched_ppl:.2f}  KL(base||patched)={patched_kl:.4f}")

    # inspect learned patch
    inspect_patch()

    # optional: run sparsity experiment (comment out if too slow)
    sparsity_ablation()


if __name__ == "__main__":
    main()
