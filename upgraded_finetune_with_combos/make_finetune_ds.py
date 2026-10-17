"""Fine-tune dataset. Apriori top-k (hit@k) candidates are context; the clean expected outcome is the answer.
Every prompt also gets a CART SUMMARY (label counts across the cart) unless --no-cart-context.
  python make_finetune_ds.py                                   # candidates shown in TRAIN prompts only (half of them), test = cart-only
  python make_finetune_ds.py --candidates both                 # re-ranker: candidates in train AND test (needs mining at inference)
  python make_finetune_ds.py --candidates none                 # plain cart -> answer
Works on any DS1 (context ids -> outcome id) + DS2 (id -> description)."""
import argparse, json, random
from pathlib import Path
from apriori_core import HERE, load_json, mine
from prompt_format import build_prompt, DEFAULT_FIELDS

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="ds1_train.json"); ap.add_argument("--test", default="ds1_test.json")
ap.add_argument("--ds2", default="ds2_descriptions.json"); ap.add_argument("--out-dir", default="finetune")
ap.add_argument("--fields", default=",".join(DEFAULT_FIELDS), help="DS2 fields shown per item (id + name always shown)")
ap.add_argument("--candidates", choices=["train-only", "both", "none"], default="train-only",
                help="where the Apriori top-k is shown: train-only = train prompts only, test/inference prompts are cart-only | both | none")
ap.add_argument("--cand-dropout", type=float, default=0.5, help="share of TRAIN rows whose candidates are hidden, so the model also learns the cart-only format")
ap.add_argument("--top-k", type=int, default=3, help="Apriori candidates (the hit@k context)")
ap.add_argument("--target", choices=["auto", "expected", "observed"], default="auto",
                help="expected = clean label from meta.expected (deterministic); observed = noisy DS1 outcome; auto = expected if present")
ap.add_argument("--on-miss", choices=["drop", "inject", "keep"], default="drop",
                help="train rows that SHOW candidates but the answer is not among them: drop | inject the answer in place of the lowest-ranked | keep")
ap.add_argument("--order", choices=["shuffle", "rank"], default="shuffle")
ap.add_argument("--answer", choices=["id_name", "id", "name"], default="id_name")
ap.add_argument("--format", choices=["text", "chat"], default="text"); ap.add_argument("--instruction", default="", help="'' = automatic")
ap.add_argument("--folds", type=int, default=5, help="train candidates come from rules mined on the OTHER folds (1 = in-sample, over-optimistic)")
ap.add_argument("--group-key", default="combo", help="meta key whose values are kept together in a fold (falls back to per-row)")
ap.add_argument("--min-support", type=float, default=0.004); ap.add_argument("--min-conf", type=float, default=0.3)
ap.add_argument("--cart-context", action=argparse.BooleanOptionalAction, default=True,
                help="add a CART SUMMARY line: label counts across the cart (--no-cart-context to turn off)")
ap.add_argument("--summary-field", default="labels", help="list-valued DS2 field summarised across the cart")
ap.add_argument("--field-dropout", type=float, default=0.0,
                help="share of TRAIN rows where some description fields are hidden (robustness to thinner catalogs)")
ap.add_argument("--protect-fields", default="labels", help="fields field-dropout never hides")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

train, test = load_json(a.train), load_json(a.test)
ds2 = {d["item_id"]: d for d in load_json(a.ds2)}
fields = [f.strip() for f in a.fields.split(",") if f.strip()]
protect = {f.strip() for f in a.protect_fields.split(",")}
rules, _ = mine(train, a.min_support, a.min_conf)                      # mined on TRAIN only
print(f"{len(rules)} rules mined from {len(train)} train rows | top-{a.top_k} candidates | mode: {a.candidates}"
      + (f", {a.cand_dropout:.0%} of train rows hide them" if a.candidates != "none" else ""))

def row_rules_for_train():
    if a.folds <= 1: return [rules] * len(train)
    keys = [str(r.get("meta", {}).get(a.group_key, i)) for i, r in enumerate(train)]
    groups = sorted(set(keys)); random.Random(a.seed).shuffle(groups)
    fold_of = {g: i % a.folds for i, g in enumerate(groups)}
    per_fold = [mine([r for r, k in zip(train, keys) if fold_of[k] != f], a.min_support, a.min_conf)[0] for f in range(a.folds)]
    return [per_fold[fold_of[k]] for k in keys]

def build(rows, split, row_rules, show, label):
    out, dropped, inj, inc, top1, shown = [], 0, 0, 0, 0, 0
    for idx, r in enumerate(rows):
        meta = r.get("meta", {})
        target = meta["expected"] if (a.target == "expected" or (a.target == "auto" and "expected" in meta)) else r["outcome"]
        row_fields, hidden = fields, []
        if split == "train" and a.field_dropout > 0:
            frng = random.Random(a.seed * 7919 + idx)
            if frng.random() < a.field_dropout:
                hidden = [f for f in fields if f not in protect and frng.random() < 0.5]
                row_fields = [f for f in fields if f not in hidden]
        args = (r["context"], ds2, row_rules[idx], a.top_k, row_fields, a.instruction or None, a.order, a.seed + idx)
        kw = dict(cart_context=a.cart_context, summary_field=a.summary_field)
        prompt, ranked = build_prompt(*args, show_candidates=show[idx], **kw)
        hit = target in ranked; injected = False
        if split == "train" and show[idx] and not hit:
            if a.on_miss == "drop": dropped += 1; continue
            if a.on_miss == "inject":
                ranked = ranked[:a.top_k - 1] + [target]
                prompt, _ = build_prompt(*args, candidates=ranked, show_candidates=True, **kw); hit = injected = True; inj += 1
        t = ds2[target]
        comp = {"id_name": f"{target} | {t['name']}", "id": target, "name": t["name"]}[a.answer]
        inc += hit; top1 += bool(ranked) and ranked[0] == target; shown += show[idx]
        m = {**meta, "split": split, "outcome_id": target, "context_ids": r["context"], "candidates": ranked,
             "outcome_in_candidates": hit, "injected": injected, "shows_candidates": show[idx],
             "apriori_top1": ranked[0] if ranked else None, "hidden_fields": hidden}
        rec = ({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": comp}]}
               if a.format == "chat" else {"prompt": prompt, "completion": comp})
        out.append({**rec, "meta": m})
    n = max(len(out), 1)
    print(f"{label}: {len(out)} rows | candidates shown in {shown} | dropped {dropped}, injected {inj} | "
          f"Apriori top-3 has answer: {inc/n:.1%} | Apriori top-1 alone: {top1/n:.1%} | avg {sum(len(json.dumps(x)) for x in out)//n} chars")
    return out

rr = random.Random(a.seed + 1)
train_show = [a.candidates != "none" and rr.random() >= a.cand_dropout for _ in train]
jobs = [(train, "train", row_rules_for_train(), train_show, "finetune_train.jsonl"),
        (test, "test", [rules] * len(test), [a.candidates == "both"] * len(test), "finetune_test.jsonl")]
if a.candidates == "train-only":      # same test rows WITH candidates, for an A/B check
    jobs.append((test, "test", [rules] * len(test), [True] * len(test), "finetune_test_with_candidates.jsonl"))
out_dir = Path(a.out_dir) if Path(a.out_dir).is_absolute() else HERE / a.out_dir
out_dir.mkdir(exist_ok=True)
json.dump({"fields": fields, "cart_context": a.cart_context, "summary_field": a.summary_field, "order": a.order},
          open(out_dir / "prompt_config.json", "w"), indent=1)      # inference reads this, so the format cannot drift
for rows, split, rrules, show, fname in jobs:
    data = build(rows, split, rrules, show, fname)
    with open(out_dir / fname, "w") as f:
        for x in data: f.write(json.dumps(x) + "\n")
    if fname == "finetune_test.jsonl": sample = next(x for x in data if x["meta"].get("kind") == "mixed")
print("\n----- sample test prompt (what inference will look like) -----\n" + (sample.get("prompt") or sample["messages"][0]["content"])
      + "\n\n>>> " + (sample.get("completion") or sample["messages"][1]["content"]))
