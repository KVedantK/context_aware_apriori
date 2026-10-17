"""LoRA fine-tune (HF transformers + peft) on finetune/finetune_train.jsonl, then hit@1 on finetune_test.jsonl.
  pip install -U transformers peft datasets accelerate
  (runs on CUDA, Apple-silicon MPS or CPU; on a Mac prefer train_mlx.py, it is faster and lighter)
  python train_lora.py                                        # Qwen2.5-0.5B-Instruct
  python train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --bs 4 --grad-accum 4
"""
import argparse, json, re, torch
from collections import defaultdict
from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--train", default="finetune/finetune_train.jsonl"); ap.add_argument("--test", default="finetune/finetune_test.jsonl")
ap.add_argument("--out", default="out_lora"); ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--grad-accum", type=int, default=2); ap.add_argument("--max-len", type=int, default=1024)
ap.add_argument("--r", type=int, default=16); ap.add_argument("--eval-n", type=int, default=0, help="0 = whole test set"); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--hub-repo", default="", help="push the adapter to this HF repo (e.g. you/qwen-apriori-lora) right after training")
ap.add_argument("--pause-space", action="store_true", help="when running in an HF Space: pause it on exit so billing stops")
a = ap.parse_args()
if a.pause_space:
    import atexit, os
    def _pause():
        try:
            from huggingface_hub import HfApi; HfApi().pause_space(os.environ["SPACE_ID"])
        except Exception as e: print("could not pause space:", e)
    atexit.register(_pause)                     # runs on normal exit AND on crashes
rd = lambda p: [json.loads(l) for l in open(p if Path(p).exists() else HERE / p)]
train, test = rd(a.train), rd(a.test)
if a.eval_n: test = test[:a.eval_n]

tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
def encode(r):                       # loss on the completion only
    p = tok(r["prompt"], add_special_tokens=False).input_ids[-(a.max_len - 32):]
    c = tok(" " + r["completion"], add_special_tokens=False).input_ids + [tok.eos_token_id]
    return {"input_ids": p + c, "attention_mask": [1] * (len(p) + len(c)), "labels": [-100] * len(p) + c}
data = [encode(r) for r in train]

dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
bf16 = dev == "cuda" and torch.cuda.is_bf16_supported(); fp16 = dev == "cuda" and not bf16
dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)   # Mac/CPU: fp32 (stable)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev)
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
for p in model.parameters():
    if p.requires_grad: p.data = p.data.float()          # fp32 adapters (needed for fp16 training on T4)
model.print_trainable_parameters()

Trainer(model=model, train_dataset=data, data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True),
        args=TrainingArguments(output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=a.bs,
                               gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, lr_scheduler_type="cosine",
                               warmup_ratio=0.05, logging_steps=20, save_strategy="no", bf16=bf16, fp16=fp16,
                               report_to="none", dataloader_pin_memory=(dev == "cuda"), seed=a.seed, remove_unused_columns=False)).train()
model.save_pretrained(a.out); tok.save_pretrained(a.out)
if a.hub_repo: model.push_to_hub(a.hub_repo, private=True); tok.push_to_hub(a.hub_repo, private=True)   # Space disks are ephemeral

# ---------------- eval: first token of the answer must equal the expected item id ----------------
model.eval(); model.config.use_cache = True
stats, preds = defaultdict(lambda: [0, 0]), []
for i in range(0, len(test), 16):
    batch = test[i:i + 16]
    enc = tok([r["prompt"] for r in batch], return_tensors="pt", padding=True, truncation=True, max_length=a.max_len).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
    for r, o in zip(batch, out):
        text = tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        got, exp = (text.split() or [""])[0], r["meta"]["outcome_id"]
        preds.append({"got": text, "expected": r["completion"], "hit": got == exp, "meta": r["meta"]})
        for g in ["ALL"] + [f"{k}: {r['meta'][k]}" for k in ("family", "kind") if k in r["meta"]] + \
                 [f"answer in candidates: {r['meta']['outcome_in_candidates']}"] * ('outcome_in_candidates' in r["meta"]):
            stats[g][0] += 1; stats[g][1] += got == exp
        stats["BASELINE apriori top-1"][0] += 1; stats["BASELINE apriori top-1"][1] += r["meta"].get("apriori_top1") == exp
print(f"\n{'group':34s} {'n':>4s} {'hit@1':>7s}")
for g in ["ALL"] + sorted(g for g in stats if g != "ALL"): print(f"{g:34s} {stats[g][0]:4d} {stats[g][1]/stats[g][0]:7.1%}")
ceil = [r["meta"].get("outcome_in_candidates") for r in test]
if any(c is not None for c in ceil): print(f"\nceiling (answer present in candidate list): {sum(bool(c) for c in ceil)/len(ceil):.1%}")
with open(Path(a.out) / "test_predictions.jsonl", "w") as f:
    for p in preds: f.write(json.dumps(p) + "\n")
open(Path(a.out) / "results.txt", "w").write("\n".join(f"{g}: n={stats[g][0]} hit@1={stats[g][1]/stats[g][0]:.3f}" for g in sorted(stats)))
if a.hub_repo:                          # keep results too: a paused Space loses its disk
    from huggingface_hub import HfApi
    for fn in ("test_predictions.jsonl", "results.txt"):
        HfApi().upload_file(path_or_fileobj=str(Path(a.out) / fn), path_in_repo=fn, repo_id=a.hub_repo)
