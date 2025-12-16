import math
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForMaskedLM

# =========================
# CONFIG
# =========================
MODEL_NAME = "vinai/phobert-base"
CKPT_PATH = "/mnt/d/OCR/finetune_step_2002.pt"   # chỉnh đúng path của bạn
MAX_LEN = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD TOKENIZER & MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

MASK = tokenizer.mask_token_id
PAD  = tokenizer.pad_token_id
CLS  = tokenizer.cls_token
SEP  = tokenizer.sep_token

# =========================
# PREPARE CONDITIONAL INPUT
# =========================
@torch.no_grad()
def prepare_conditional(seq_len, en_text):
    prompt = f"{CLS} {en_text} {SEP}"
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    x = torch.full((1, seq_len), MASK, dtype=torch.long, device=device)
    mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
    attn = torch.ones((1, seq_len), dtype=torch.long, device=device)

    Lp = min(len(ids), seq_len)
    x[0, :Lp] = torch.tensor(ids[:Lp], device=device)
    mask[0, :Lp] = False

    return x, mask, attn, mask.clone()

# =========================
# CLEAN OUTPUT (SAFE)
# =========================
def clean_output(ids, mask_init):
    region = mask_init[0].cpu().tolist()
    out_ids = [tid for tid, r in zip(ids, region) if r]

    # chỉ cắt </s> nếu nó KHÔNG phải token đầu
    if tokenizer.sep_token_id in out_ids[1:]:
        out_ids = out_ids[:out_ids.index(tokenizer.sep_token_id)]

    # fallback nếu vẫn rỗng
    if len(out_ids) == 0:
        out_ids = [tid for tid, r in zip(ids, region)]

    return tokenizer.decode(out_ids, skip_special_tokens=True).strip()

# =========================
# DIFFUSION TRANSLATION
# =========================
@torch.no_grad()
def diffusion_translate(en_text, num_steps=100, strategy="random", seq_len=160):

    input_tokens, mask, attn, mask_init = prepare_conditional(seq_len, en_text)
    times = torch.linspace(1, 0, num_steps + 1, device=device)

    for _, s in zip(times[:-1], times[1:]):

        logits = model(input_tokens, attention_mask=attn).logits
        probs  = torch.softmax(logits, dim=-1)
        pred   = probs.argmax(dim=-1)

        # fill current masked positions
        input_tokens[mask] = pred[mask]

        region = mask_init[0]
        idx = torch.where(region)[0]
        n = idx.numel()

        if n == 0:
            break

        # ===== RANDOM STRATEGY =====
        if strategy == "random":
            nun = int(math.floor(n * (1.0 - float(s))))
            nun = max(0, min(nun, n))

            if nun == 0:
                mask[:] = False
                continue

            perm = torch.randperm(n, device=device)
            keep_idx = idx[perm[:nun]]

            new_mask = torch.zeros_like(mask)
            new_mask[0, keep_idx] = True
            mask = new_mask

            input_tokens[mask] = MASK

        # ===== LOW-CONFIDENCE STRATEGY =====
        elif strategy == "lowconf":
            conf = torch.gather(probs, 2, pred.unsqueeze(-1)).squeeze(-1)
            conf[0, input_tokens[0] == PAD] = 0.0

            k = int(n * float(s))
            k = max(0, min(k, n))

            if k > 0:
                _, local_idx = torch.topk(conf[0, idx], k, largest=False)
                remask_idx = idx[local_idx]

                new_mask = torch.zeros_like(mask)
                new_mask[0, remask_idx] = True
                mask = new_mask

                input_tokens[mask] = MASK
            else:
                mask[:] = False

    # ===== FINAL UNMASK (BẮT BUỘC) =====
    if mask.any():
        logits = model(input_tokens, attention_mask=attn).logits
        pred = logits.argmax(dim=-1)
        input_tokens[mask] = pred[mask]
        mask[:] = False

    return clean_output(input_tokens[0].tolist(), mask_init)

# =========================
# GRADIO UI
# =========================
def translate_ui(text, steps, strategy):
    if not text.strip():
        return "❌ Please enter an English sentence."
    return diffusion_translate(
        text,
        num_steps=int(steps),
        strategy=strategy,
        seq_len=160
    )

demo = gr.Blocks(title="EN → VI | PhoBERT Diffusion")

with demo:
    gr.Markdown("""
    # 🌐 English → Vietnamese Translation  
    **PhoBERT + Diffusion-style MLM**  
    """)

    with gr.Row():
        with gr.Column():
            en_input = gr.Textbox(
                label="English Input",
                lines=4,
                placeholder="Enter English sentence..."
            )

            strategy = gr.Radio(
                ["random", "lowconf"],
                value="lowconf",
                label="Sampling Strategy"
            )

            steps = gr.Slider(
                minimum=20,
                maximum=300,
                step=10,
                value=120,
                label="Number of Diffusion Steps"
            )

            btn = gr.Button("Translate", variant="primary")

        with gr.Column():
            vi_output = gr.Textbox(
                label="Vietnamese Output",
                lines=6
            )

    btn.click(
        translate_ui,
        inputs=[en_input, steps, strategy],
        outputs=vi_output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
