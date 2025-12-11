import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file
from datasets import load_dataset
import datetime

from config import (
    MODEL_NAME,
    REVISION,
    DTYPE,
    BASE_PATH,
    BROKEN_PATH,
    PATCH_PATH,
    SUPERWEIGHTS,
    MAX_LEN,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
)
from patching import apply_patch_to_olmo, save_patch_safetensors


def load_model_from(path):
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
    return model


def get_batches(tokenizer, texts, batch_size=2, max_len=128):
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
            # simple left-pad
            maxL = max(b["input_ids"].size(1) for b in batch)
            ids = []
            attn = []
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            for b in batch:
                L = b["input_ids"].size(1)
                pad = maxL - L
                if pad > 0:
                    pad_tensor = torch.full((1, pad), pad_id, dtype=torch.long)
                    ids.append(torch.cat([b["input_ids"], pad_tensor], dim=1))
                else:
                    ids.append(b["input_ids"])
                attn.append((ids[-1] != pad_id).long())
            input_ids = torch.cat(ids, dim=0)
            attention_mask = torch.cat(attn, dim=0)
            yield {"input_ids": input_ids, "attention_mask": attention_mask}
            batch = []


def main():
    dtype = getattr(torch, DTYPE)

    # Teacher = base model
    teacher = load_model_from(BASE_PATH)
    teacher.eval()

    # Student = broken model
    broken = load_model_from(BROKEN_PATH)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Apply patch (only first superweight's row for now)
    sw = SUPERWEIGHTS[0]
    layer_idx = sw["layer"]
    row = sw["row"]
    col = sw["col"]  # we don't use 'col' directly here, but it's the ablated entry
    print(f"Applying patch to layer {layer_idx}, row {row}")

    patched_model, patch_module = apply_patch_to_olmo(broken, layer_idx, row)
    patched_model.train()

    # Only delta_row is trainable
    optimizer = optim.AdamW([patch_module.delta_row], lr=LEARNING_RATE)

    # Tiny dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:2%]")
    texts = [x["text"] for x in ds]

    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(get_batches(tokenizer, texts, BATCH_SIZE, MAX_LEN)):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            student_out = patched_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Shift for next-token prediction
            t_logits = teacher_out.logits[:, :-1, :]
            s_logits = student_out.logits[:, :-1, :]

            t_probs = F.softmax(t_logits, dim=-1)
            s_log_probs = F.log_softmax(s_logits, dim=-1)

            loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Epoch {epoch} step {i} loss {loss.item():.4f}")
            print(f"[{datetime.datetime.now()}] Epoch {epoch} step {i} loss {loss.item():.4f}")

            # for your Mac: bail early if this is too slow
            if i > 200: 
                break

    # Save patch as safetensors
    # We need the *broken* state dict as base for shapes.
    broken_state = load_file(BROKEN_PATH)
    with torch.no_grad():
        delta_row = patch_module.delta_row.detach().cpu()
    save_patch_safetensors(broken_state, layer_idx, row, delta_row)

if __name__ == "__main__":
    main()