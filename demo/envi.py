import torch
import gradio as gr
from model_iwslt import (
    Transformer,
    SimpleTokenizer,
    make_src_mask,
    make_tgt_mask
)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD TRAIN DATA (để build tokenizer)
# =========================
DATA_PATH = "/mnt/d/OCR/data"   # ⚠️ chỉnh đúng path WSL của bạn

train_en = open(f"{DATA_PATH}/train.en", encoding="utf8").read().splitlines()
train_vi = open(f"{DATA_PATH}/train.vi", encoding="utf8").read().splitlines()

src_tok = SimpleTokenizer()
tgt_tok = SimpleTokenizer()
src_tok.fit(train_en)
tgt_tok.fit(train_vi)

print("✅ Tokenizer built")
print("SRC vocab:", src_tok.vocab_size_())
print("TGT vocab:", tgt_tok.vocab_size_())

# =========================
# LOAD MODEL
# =========================
model = Transformer(
    src_tok.vocab_size_(),
    tgt_tok.vocab_size_(),
    d_model=256,
    N=4,
    heads=4
).to(device)

MODEL_PATH = "/mnt/d/OCR/iwslt_model_best.pt"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Model loaded successfully")

# =========================
# GREEDY DECODE
# =========================
@torch.no_grad()
def translate(sentence, max_len=80):
    model.eval()

    src_ids = torch.LongTensor(
        src_tok.encode(sentence)
    ).unsqueeze(0).to(device)

    ys = torch.LongTensor(
        [[tgt_tok.word2id[tgt_tok.BOS]]]
    ).to(device)

    for _ in range(max_len):
        out = model(
            src_ids,
            ys,
            make_src_mask(src_ids),
            make_tgt_mask(ys)
        )
        next_word = out[:, -1, :].argmax(-1).item()
        ys = torch.cat(
            [ys, torch.LongTensor([[next_word]]).to(device)],
            dim=1
        )

        if next_word == tgt_tok.word2id[tgt_tok.EOS]:
            break

    return tgt_tok.decode(ys[0].cpu().tolist())

# =========================
# GRADIO UI
# =========================
def translate_ui(text):
    if not text.strip():
        return "❌ Please enter an English sentence."
    return translate(text)

demo = gr.Blocks(title="EN → VI Transformer (From Scratch)")

with demo:
    gr.Markdown("""
    # 🌐 English → Vietnamese Translation
    **Transformer from scratch**  
    """)

    with gr.Row():
        with gr.Column():
            en_input = gr.Textbox(
                label="English Input",
                lines=4,
                placeholder="Enter English sentence..."
            )
            btn = gr.Button("Translate", variant="primary")

        with gr.Column():
            vi_output = gr.Textbox(
                label="Vietnamese Output",
                lines=4
            )

    btn.click(translate_ui, en_input, vi_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
# http://127.0.0.1:7860/