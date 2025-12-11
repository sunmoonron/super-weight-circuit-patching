import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

from config import (
    MODEL_NAME,
    REVISION,
    DTYPE,
    BROKEN_PATH,
    PATCH_PATH,
    SUPERWEIGHTS,
    MAX_LEN,
)
from patching import apply_patch_to_olmo, save_patch_safetensors

# *** CRUCIAL: SHRINK THESE FOR CPU ***
MAX_LEN = 64       # override to smaller than before
BATCH_SIZE = 1     # small to reduce per-step cost
MAX_STEPS = 30     # hard cap on total update steps


def load_broken_model():
    dtype = getattr(torch, DTYPE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=REVISION,
        torch_dtype=dtype,
        device_map=None,
    )
    state = load_file(BROKEN_PATH)
    model.load_state_dict(state)
    model.eval()
    return model


def get_training_texts():
    # Build a tiny dataset focusing on your failure cases
    return [
        "Summer is hot. Winter is",
        "Paris is in France. Tokyo is in",
        "2, 4, 6, 8,",
        "The capital of Canada is",
        'He opened the quote " and then wrote Hello. To close it he should type',
        "List of numbers: [1, 2, 3",
        "In Python, to close a string started with ' you end with ",
        "The opposite of big is",
        "A cat is an animal. A rose is a",
        "Complete the analogy: day : night :: up :",
    ] * 3  # repeat a bit for more steps


def make_batches(tokenizer, texts, batch_size=1, max_len=64):
    batch = []
    for t in texts:
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
            # Simple padding for batch_size=1 is trivial, but let's keep it generic
            maxL = max(b["input_ids"].size(1) for b in batch)
            ids_list = []
            attn_list = []
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


def main():
    dtype = getattr(torch, DTYPE)

    # 1. Load broken model
    broken = load_broken_model()

    # 2. Apply patch module (only first superweight row)
    sw = SUPERWEIGHTS[0]
    layer_idx = sw["layer"]
    row = sw["row"]
    col = sw["col"]
    print(f"Applying patch to layer {layer_idx}, row {row}, col {col}")

    patched_model, patch_module = apply_patch_to_olmo(broken, layer_idx, row)
    patched_model.train()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    texts = get_training_texts()

    optimizer = optim.AdamW([patch_module.delta_row], lr=1e-3)

    global_step = 0
    for batch in make_batches(tokenizer, texts, batch_size=BATCH_SIZE, max_len=MAX_LEN):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Shift for next-token prediction
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        attention_mask = attention_mask[:, :-1].contiguous()

        out = patched_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = out.logits  # [B, T, V]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=tokenizer.pad_token_id or -100,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {global_step} loss {loss.item():.4f}")

        global_step += 1
        if global_step >= MAX_STEPS:
            break

    # Save the patch
    broken_state = load_file(BROKEN_PATH)
    with torch.no_grad():
        delta_row = patch_module.delta_row.detach().cpu()
    save_patch_safetensors(broken_state, layer_idx, row, delta_row)


if __name__ == "__main__":
    main()
