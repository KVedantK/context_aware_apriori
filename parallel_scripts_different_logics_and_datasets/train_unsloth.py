"""LoRA fine-tune with Unsloth on finetune/finetune_train.jsonl, then hit@1 on finetune_test.jsonl.
  pip install unsloth            (Colab/Linux + NVIDIA GPU)
  python train_unsloth.py                                          # unsloth/Qwen2.5-0.5B-Instruct
  python train_unsloth.py --model unsloth/Qwen2.5-1.5B-Instruct    # add --4bit to save VRAM
"""
import argparse, json, torch
from unsloth import FastLanguageModel          # must be imported before transformers
from collections import defaultdict
from pathlib import Path
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="unsloth/Qwen2.5-0.5B-Instruct"); ap.add_argument("--4bit", dest="four_bit", action="store_true")
ap.add_argument("--train", default="finetune/finetune_train.jsonl"); ap.add_argument("--test", default="finetune/finetune_test.jsonl")
ap.add_argument("--out", default="out_unsloth"); ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--grad-accum", type=int, default=2); ap.add_argument("--max-len", type=int, default=1024)
ap.add_argument("--r", type=int, default=16); ap.add_argument("--eval-n", type=int, default=0, help="0 = whole test set"); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
rd = lambda p: [json.loads(l) for l in open(p if Path(p).exists() else HERE / p)]
train, test = rd(a.train), rd(a.test)
if a.eval_n: test = test[:a.eval_n]

model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=a.max_len, dtype=None, load_in_4bit=a.four_bit)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = FastLanguageModel.get_peft_model(model, r=a.r, lora_alpha=2 * a.r, lora_dropout=0, bias="none", random_state=a.seed,
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                         use_gradient_checkpointing="unsloth")
def encode(r):                       # loss on the completion only
    p = tok(r["prompt"], add_special_tokens=False).input_ids[-(a.max_len - 32):]
    c = tok(" " + r["completion"], add_special_tokens=False).input_ids + [tok.eos_token_id]
    return {"input_ids": p + c, "attention_mask": [1] * (len(p) + len(c)), "labels": [-100] * len(p) + c}
data = [encode(r) for r in train]

bf16 = torch.cuda.is_bf16_supported()
Trainer(model=model, train_dataset=data, data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True),
        args=TrainingArguments(output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=a.bs,
                               gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, lr_scheduler_type="cosine",
                               warmup_ratio=0.05, logging_steps=20, save_strategy="no", bf16=bf16, fp16=not bf16, optim="adamw_8bit",
                               report_to="none", seed=a.seed, remove_unused_columns=False)).train()
model.save_pretrained(a.out); tok.save_pretrained(a.out)

# ---------------- eval: first token of the answer must equal the expected item id ----------------
FastLanguageModel.for_inference(model)
stats, preds = defaultdict(lambda: [0, 0]), []
for i in range(0, len(test), 16):
    batch = test[i:i + 16]
    enc = tok([r["prompt"] for r in batch], return_tensors="pt", padding=True, truncation=True, max_length=a.max_len).to("cuda")
    out = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    for r, o in zip(batch, out):
        text = tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        got, exp = (text.split() or [""])[0], r["meta"]["outcome_id"]
        preds.append({"got": text, "expected": r["completion"], "hit": got == exp, "meta": r["meta"]})
        for g in ["ALL"] + [f"{k}: {r['meta'][k]}" for k in ("family", "kind") if k in r["meta"]]:
            stats[g][0] += 1; stats[g][1] += got == exp
print(f"\n{'group':34s} {'n':>4s} {'hit@1':>7s}")
for g in ["ALL"] + sorted(g for g in stats if g != "ALL"): print(f"{g:34s} {stats[g][0]:4d} {stats[g][1]/stats[g][0]:7.1%}")
ceil = [r["meta"].get("outcome_in_candidates") for r in test]
if any(c is not None for c in ceil): print(f"\nceiling (answer present in candidate list): {sum(bool(c) for c in ceil)/len(ceil):.1%}")
with open(Path(a.out) / "test_predictions.jsonl", "w") as f:
    for p in preds: f.write(json.dumps(p) + "\n")