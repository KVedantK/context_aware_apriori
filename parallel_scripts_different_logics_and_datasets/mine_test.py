"""Traditional Apriori on DS1-train (ids only; DS2 only supplies names for printing), evaluated on DS1-test.
  python mine_test.py [--min-support 0.004 --min-conf 0.3 --group-by family,kind --show-misses 8]
Needs only: rows with "context" (list of ids) and "outcome" (id); DS2 rows with "item_id" and "name".
Optional "meta" fields on test rows are used for the per-group tables (--group-by); without them you just get ALL."""
import argparse
from collections import defaultdict
from apriori_core import load_json, mine, predict
ap = argparse.ArgumentParser()
ap.add_argument("--train", default="ds1_train.json"); ap.add_argument("--test", default="ds1_test.json"); ap.add_argument("--ds2", default="ds2_descriptions.json")
ap.add_argument("--min-support", type=float, default=0.004); ap.add_argument("--min-conf", type=float, default=0.3)
ap.add_argument("--group-by", default="family,kind", help="comma-separated meta keys to break results down by ('' for none)")
ap.add_argument("--show-misses", type=int, default=8)
a = ap.parse_args()

train, test = load_json(a.train), load_json(a.test)
nm = {d["item_id"]: d["name"] for d in load_json(a.ds2)}
rules, n_freq = mine(train, a.min_support, a.min_conf)
print(f"train={len(train)} test={len(test)} | frequent itemsets={n_freq} | outcome rules={len(rules)} "
      f"(min_support={a.min_support}, min_confidence={a.min_conf})\n\nTop rules:")
for ant, c, sup, conf, lift in rules[:8]:
    print(f"  {' + '.join(sorted(nm[i] for i in ant)):52s} => {nm[c]:22s} support={sup:.3f} confidence={conf:.2f}")

keys = [k.strip() for k in a.group_by.split(",") if k.strip()]
res, stats = [], defaultdict(lambda: [0, 0, 0, 0])
for r in test:
    top = predict(rules, r["context"]); ids = [t[0] for t in top]
    h1, h3 = bool(ids) and ids[0] == r["outcome"], r["outcome"] in ids
    groups = ["ALL"] + [f"{k}: {r['meta'][k]}" for k in keys if r.get("meta", {}).get(k) is not None]
    for g in groups:
        s = stats[g]; s[0] += 1; s[1] += h1; s[2] += h3; s[3] += bool(ids)
    res.append((r, top, h1))

print(f"\n{'group':34s} {'n':>4s} {'hit@1':>7s} {'hit@3':>7s} {'answered':>9s}")
for g in ["ALL"] + sorted(g for g in stats if g != "ALL"):
    n, h1, h3, cov = stats[g]
    print(f"{g:34s} {n:4d} {h1/n:7.1%} {h3/n:7.1%} {cov/n:9.1%}")

print("\nSample misses:")
for r, top, h1 in [x for x in res if not x[2]][:a.show_misses]:
    tag = ",".join(str(v) for v in r.get("meta", {}).values() if isinstance(v, str) and not v.startswith("OB_")) or "-"
    print(f"  [{tag}] {' + '.join(nm[i] for i in r['context'])}\n      expected: {nm[r['outcome']]}"
          f"\n      got     : {', '.join(f'{nm[t[0]]} ({t[1]:.2f})' for t in top) or '-- no rule --'}")
