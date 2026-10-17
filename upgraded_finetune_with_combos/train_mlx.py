"""Mac (Apple silicon) LoRA fine-tune with Apple's MLX (mlx-lm), then hit@1 on finetune_test.jsonl.
  pip install -U "mlx-lm[train]"
  python train_mlx.py                                       # Qwen/Qwen2.5-0.5B-Instruct
  python train_mlx.py --model Qwen/Qwen2.5-1.5B-Instruct
  python train_mlx.py --eval-n 50 --epochs 1                # quick smoke test
"""
import argparse, json, math, subprocess, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--train", default="finetune/finetune_train.jsonl"); ap.add_argument("--test", default="finetune/finetune_test.jsonl")
ap.add_argument("--out", default="out_mlx"); ap.add_argument("--data-dir", default="mlx_data")
ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--num-layers", type=int, default=16, help="layers to LoRA-tune (-1 = all, if your mlx-lm supports it)")
ap.add_argument("--max-len", type=int, default=1024); ap.add_argument("--n-valid", type=int, default=100)
ap.add_argument("--eval-n", type=int, default=0, help="0 = whole test set"); ap.add_argument("--skip-train", action="store_true")
ap.add_argument("--no-chat-template", action="store_true", help="eval prompts without the chat template")
a = ap.parse_args()
rd = lambda p: [json.loads(l) for l in open(p if Path(p).exists() else HERE / p)]
train, test = rd(a.train), rd(a.test)
if a.eval_n: test = test[:a.eval_n]

# ---- 1. data dir for mlx_lm.lora: train.jsonl + valid.jsonl, {"prompt","completion"} only ----
data_dir = HERE / a.data_dir; data_dir.mkdir(exist_ok=True)
valid, tr = train[-a.n_valid:], train[:-a.n_valid]
for name, rows in (("train", tr), ("valid", valid)):
    with open(data_dir / f"{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}) + "\n")

# ---- 2. train ----
if not a.skip_train:
    iters = math.ceil(a.epochs * len(tr) / a.bs)
    cmd = [sys.executable, "-m", "mlx_lm.lora", "--model", a.model, "--train", "--data", str(data_dir), "--adapter-path", a.out,
           "--iters", str(iters), "--batch-size", str(a.bs), "--learning-rate", str(a.lr), "--num-layers", str(a.num_layers),
           "--max-seq-length", str(a.max_len), "--mask-prompt", "--steps-per-eval", "100", "--save-every", "200"]
    print(" ".join(cmd)); subprocess.run(cmd, check=True)

# ---- 3. eval: first token of the answer must equal the expected item id ----
from mlx_lm import load, generate
model, tok = load(a.model, adapter_path=a.out)
fmt = (lambda p: p) if a.no_chat_template or not getattr(tok, "chat_template", None) else \
      (lambda p: tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False))
stats, preds = defaultdict(lambda: [0, 0]), []
for r in test:
    text = generate(model, tok, prompt=fmt(r["prompt"]), max_tokens=16).strip()
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
Path(a.out).mkdir(exist_ok=True)
with open(Path(a.out) / "test_predictions.jsonl", "w") as f:
    for p in preds: f.write(json.dumps(p) + "\n")
