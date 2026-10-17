"""Spec-driven dataset builder.  python build_ds.py [--n-train 1800 --n-test 400 --seed 42]
Writes ds2_descriptions.json, ds1_train.json, ds1_test.json next to this file."""
import argparse, itertools, json, math, random
from collections import Counter
from pathlib import Path
import importlib, sys

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--n-train", type=int, default=1800); ap.add_argument("--n-test", type=int, default=400)
ap.add_argument("--spec", default="world_spec", help="python module holding ITEMS, FAMILIES, FILLERS, COLD_START, SPECIAL_LABELS")
ap.add_argument("--id-prefix", default="OB_"); ap.add_argument("--id-start", type=int, default=1001)
ap.add_argument("--mainstream-weight", type=float, default=2, help="how much more often items without special labels are bought")
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--holdout", type=float, default=0.25, help="share of slot-combos kept out of train")
ap.add_argument("--seen-frac", type=float, default=0.10, help="share of test rows that reuse train combos (control)")
ap.add_argument("--noise", type=float, default=0.06, help="train only: shopper buys the default instead of the rule's outcome")
a = ap.parse_args(); rng = random.Random(a.seed)
sys.path.insert(0, str(HERE)); W = importlib.import_module(a.spec)

# ---- DS2 ----
ds2 = [{"item_id": f"{a.id_prefix}{a.id_start + k}", "name": n, "category": c, "labels": l.split(), "shelf_life_days": s, "storage": st, "common_uses": u}
       for k, (n, c, l, s, st, u) in enumerate(W.ITEMS)]
ID = {d["name"]: d["item_id"] for d in ds2}; LAB = {d["name"]: set(d["labels"]) for d in ds2}
SPECIAL = set(getattr(W, "SPECIAL_LABELS", ()))
FILLER_IDS = {ID[n] for n in getattr(W, "FILLERS", [])}
weight = lambda n: 1 if LAB[n] & SPECIAL else a.mainstream_weight
for fam in W.FAMILIES.values():
    for slot in fam["slots"]:
        for n in slot: assert n in ID, n
    for o, *_ in fam["rules"]: assert o in ID, o

def expected(names, fam):
    labs = [LAB[n] for n in names]
    for outcome, label, k, forbid in fam["rules"]:
        if sum(label in l for l in labs) >= k and not any(f in l for l in labs for f in forbid):
            return outcome
    return fam["default"]

def make_row(combo, fam, kind, noise):
    out = expected(combo, fam)
    if out != fam["default"] and rng.random() < noise: out = fam["default"]
    F = getattr(W, "FILLERS", []); n_f = 2 if rng.random() < 0.10 else (1 if rng.random() < 0.39 else 0)
    ctx = list(combo) + rng.sample(F, min(n_f, len(F)))
    rng.shuffle(ctx)
    return {"context": [ID[x] for x in ctx], "outcome": ID[out],
            "meta": {"family": FAM_NAME, "kind": kind, "expected": ID[expected(combo, fam)], "combo": "|".join(sorted(ID[x] for x in combo))}}

train, test = [], []
fams = list(W.FAMILIES.items())
for fi, (FAM_NAME, fam) in enumerate(fams):
    combos = list(itertools.product(*fam["slots"]))
    cold = [c for c in combos if any(x in getattr(W, "COLD_START", set()) for x in c)]
    rest = [c for c in combos if c not in cold]; rng.shuffle(rest)
    k = int(len(rest) * a.holdout); held, train_c = rest[:k], rest[k:]
    for n in {x for c in rest for x in c}:                  # every non-cold item must appear in some train combo
        if not any(n in c for c in train_c):
            mv = next(c for c in held if n in c); held.remove(mv); train_c.append(mv)
    for o in {expected(c, fam) for c in combos}:            # every outcome class must be learnable from train
        if not any(expected(c, fam) == o for c in train_c):
            cand = [c for c in held if expected(c, fam) == o]
            if cand: held.remove(cand[0]); train_c.append(cand[0])
    nt = a.n_train // len(fams) + (fi < a.n_train % len(fams)); ne = a.n_test // len(fams) + (fi < a.n_test % len(fams))
    wts = [math.prod(weight(x) for x in c) for c in train_c]
    mass = Counter()
    for c, w_ in zip(train_c, wts): mass[expected(c, fam)] += w_
    wts = [w_ / mass[expected(c, fam)] ** 0.5 for c, w_ in zip(train_c, wts)]   # soften class imbalance
    for c in rng.choices(train_c, weights=wts, k=nt): train.append(make_row(c, fam, "train", a.noise))
    n_seen = round(ne * a.seen_frac); pool = held + cold
    for c in rng.choices(train_c, k=n_seen): test.append(make_row(c, fam, "seen", 0))
    for c in rng.choices(pool, k=ne - n_seen):
        test.append(make_row(c, fam, "cold" if c in cold else "mixed", 0))

rng.shuffle(train); rng.shuffle(test)
for name, obj in (("ds2_descriptions.json", ds2), ("ds1_train.json", train), ("ds1_test.json", test)):
    json.dump(obj, open(HERE / name, "w"), indent=0)
nm = {d["item_id"]: d["name"] for d in ds2}
print(f"DS2: {len(ds2)} items | DS1 train: {len(train)} rows | DS1 test: {len(test)} rows | families: {len(fams)}")
print("test kinds:", dict(Counter(r["meta"]["kind"] for r in test)))
print("train rows whose outcome was noised to the default:", sum(r["outcome"] != r["meta"]["expected"] for r in train))
tr_items = {i for r in train for i in r["context"]}
print("items never in train carts:", sorted(nm[i] for i in nm if i not in tr_items and i not in {r["outcome"] for r in train} and i not in FILLER_IDS))
for split, rows in (("train", train), ("test", test)):
    print(f"\n{split} outcome mix:")
    for fam in W.FAMILIES:
        c = Counter(nm[r["outcome"]] for r in rows if r["meta"]["family"] == fam)
        print(f"  {fam:22s} " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
