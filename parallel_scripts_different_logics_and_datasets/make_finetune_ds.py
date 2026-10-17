"""Generic fine-tune dataset builder. No families, items or labels are hard-coded: it works on any
DS1 (context ids -> outcome id) + DS2 (id -> description dict).

  python make_finetune_ds.py
  python make_finetune_ds.py --fields name,labels --top-rules 3 --candidates rules --format chat --show-stats

Per row:  CART (each item + its DS2 description) + MINED RULES (traditional Apriori matched to the cart)
          + CANDIDATES (described outcomes the model may choose from)  ->  outcome
"""
import argparse, json, random
from pathlib import Path
from apriori_core import HERE, load_json, mine, predict

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="ds1_train.json"); ap.add_argument("--test", default="ds1_test.json")
ap.add_argument("--ds2", default="ds2_descriptions.json"); ap.add_argument("--out-dir", default="finetune")
ap.add_argument("--fields", default="name,category,labels,shelf_life_days,storage,common_uses", help="DS2 fields shown per item")
ap.add_argument("--min-support", type=float, default=0.004); ap.add_argument("--min-conf", type=float, default=0.3)
ap.add_argument("--top-rules", type=int, default=4, help="matched rules shown per cart")
ap.add_argument("--candidates", choices=["siblings", "rules", "none"], default="siblings",
                help="siblings: every DS2 item in the category of a rule's outcome | rules: only rule outcomes | none")
ap.add_argument("--max-candidates", type=int, default=12)
ap.add_argument("--group-field", default="category", help="DS2 field defining siblings (items missing it get no siblings)")
ap.add_argument("--instruction", default="Suggest the best item to add to this cart. Do not Generate the outome on your own. Read the labels for extra context.")
ap.add_argument("--answer", choices=["id_name", "id", "name"], default="id_name")
ap.add_argument("--format", choices=["text", "chat"], default="text")
ap.add_argument("--show-stats", action="store_true", help="show confidence next to each rule (off by default)")
ap.add_argument("--keep-unmatched-train", action="store_true"); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

train, test = load_json(a.train), load_json(a.test)
ds2 = {d["item_id"]: d for d in load_json(a.ds2)}
by_group = {}
for d in ds2.values():
    if d.get(a.group_field) is not None: by_group.setdefault(d[a.group_field], []).append(d["item_id"])
siblings = lambda i: by_group.get(ds2[i].get(a.group_field), [i])
fields = [f.strip() for f in a.fields.split(",") if f.strip()]
rules, _ = mine(train, a.min_support, a.min_conf)          # mined on TRAIN only
print(f"{len(rules)} rules mined from {len(train)} train rows")

def describe(i):
    d = ds2[i]; parts = [i, d["name"]]
    for f in fields:
        if f != "name" and f in d:
            v = d[f]; parts.append(f"{f}: {', '.join(map(str, v)) if isinstance(v, list) else v}")
    return " | ".join(parts)

def build(rows, split, drop_unmatched):
    out, dropped, in_cand = [], 0, 0
    for idx, r in enumerate(rows):
        hits = predict(rules, r["context"], k=a.top_rules)
        if a.candidates == "none": cand = []
        else:
            ids = [h[0] for h in hits]
            if a.candidates == "siblings":
                ids = [s for i in ids for s in siblings(i)]
            cand = list(dict.fromkeys(ids))[:a.max_candidates]
            random.Random(a.seed + idx).shuffle(cand)         # no position bias
        if not hits and drop_unmatched: dropped += 1; continue
        rule_txt = "\n".join(f"{' + '.join(sorted(ds2[i]['name'] for i in h[3]))} => {ds2[h[0]]['name']}"
                             + (f"  (confidence {h[1]:.2f})" if a.show_stats else "") for h in hits) or "(no rule matched)"
        prompt = (a.instruction + "\n\nCART:\n" + "\n".join(describe(i) for i in r["context"]) +
                  f"\n\nMINED RULES:\n{rule_txt}\n\n" +
                  ("CANDIDATES:\n" + "\n".join(describe(i) for i in cand) + "\n\n" if cand else "") + "ANSWER:")
        o = ds2[r["outcome"]]
        comp = {"id_name": f"{r['outcome']} | {o['name']}", "id": r["outcome"], "name": o["name"]}[a.answer]
        in_cand += (r["outcome"] in cand) if cand else 0
        meta = {**r.get("meta", {}), "split": split, "outcome_id": r["outcome"], "context_ids": r["context"],
                "outcome_in_candidates": r["outcome"] in cand if cand else None}
        rec = ({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": comp}]}
               if a.format == "chat" else {"prompt": prompt, "completion": comp})
        out.append({**rec, "meta": meta})
    rec_pct = f"{in_cand/len(out):.1%}" if out and a.candidates != "none" else "n/a"
    print(f"{split}: {len(out)} rows (dropped {dropped} with no rule) | avg prompt {sum(len(json.dumps(x)) for x in out)//max(len(out),1)} chars"
          f" | outcome in candidate list: {rec_pct}")
    return out

out_dir = Path(a.out_dir) if Path(a.out_dir).is_absolute() else HERE / a.out_dir
out_dir.mkdir(exist_ok=True)
for rows, split, drop in ((train, "train", not a.keep_unmatched_train), (test, "test", False)):
    data = build(rows, split, drop)
    with open(out_dir / f"finetune_{split}.jsonl", "w") as f:
        for x in data: f.write(json.dumps(x) + "\n")
    if split == "test":
        s = next((x for x in data if x["meta"].get("kind") == "mixed"), data[0])
        print("\n----- sample test row -----\n" + (s.get("prompt") or s["messages"][0]["content"]) +
              "\n\n>>> " + (s.get("completion") or s["messages"][1]["content"]))
