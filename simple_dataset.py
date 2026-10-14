"""Tiny context-aware Apriori dataset: milk + bread => butter, specialised by item attributes.
Run:  python simple_dataset.py     -> writes dataset.jsonl (train + test) and prints the plain-Apriori baseline."""
import json, itertools
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ---- items: name -> (base product, attributes) ----
ITEMS = {
    # milks
    "whole milk": ("milk", []), "semi-skimmed milk": ("milk", ["low_fat"]), "skimmed milk": ("milk", ["low_fat"]),
    "lactose-free milk": ("milk", ["lactose_free"]), "soy milk": ("milk", ["plant_based"]),
    "almond milk": ("milk", ["plant_based"]), "oat milk": ("milk", ["plant_based", "low_fat"]),
    "coconut milk": ("milk", ["plant_based"]), "barista oat drink": ("milk", ["plant_based", "low_fat"]),
    # breads
    "white bread": ("bread", []), "wholemeal bread": ("bread", []), "bran loaf": ("bread", []),
    "sourdough": ("bread", ["plant_based"]), "rye bread": ("bread", ["plant_based"]),
    # butters (the answers)
    "butter": ("butter", []), "coconut butter": ("butter", ["plant_based"]),
    "low fat butter": ("butter", ["low_fat"]), "lactose-free butter": ("butter", ["lactose_free"]),
}
attrs = lambda n: set(ITEMS[n][1])

def teacher(milk, bread):
    """The labelling rule. Two plant-based signals to commit; one low-fat / lactose-free signal is enough."""
    if "plant_based" in attrs(milk) and "plant_based" in attrs(bread): return "coconut butter"
    if "low_fat" in attrs(milk) | attrs(bread): return "low fat butter"
    if "lactose_free" in attrs(milk) | attrs(bread): return "lactose-free butter"
    return "butter"

# ---- train: (milks x breads, copies) ----
TRAIN = [
    (["whole milk"], ["white bread", "wholemeal bread"], 6),                      # generic
    (["soy milk", "almond milk"], ["sourdough", "rye bread"], 4),                 # plant-based
    (["semi-skimmed milk", "skimmed milk"], ["wholemeal bread", "bran loaf"], 4),  # low fat
    (["lactose-free milk"], ["wholemeal bread", "bran loaf"], 5),                 # lactose-free
]
# ---- test: twisted, unseen combinations (+ unseen items) ----
TEST = [("oat milk", "white bread"), ("soy milk", "white bread"), ("oat milk", "sourdough"),
        ("coconut milk", "rye bread"), ("skimmed milk", "sourdough"), ("lactose-free milk", "white bread"),
        ("whole milk", "sourdough"), ("barista oat drink", "sourdough")]

def row(split, milk, bread):
    ans = teacher(milk, bread)
    fmt = lambda n: f"- {n} [{', '.join(ITEMS[n][1]) or 'no tags'}]"
    prompt = f"RULE: milk + bread => butter\nCART:\n{fmt(milk)}\n{fmt(bread)}\nSUGGEST:"
    return {"split": split, "cart": [milk, bread], "rule": "milk + bread => butter", "prompt": prompt, "completion": ans}

rows = [row("train", m, b) for ms, bs, n in TRAIN for m, b in itertools.product(ms, bs) for _ in range(n)]
rows += [row("test", m, b) for m, b in TEST]
with open("dataset.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
train, test = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "test"]
print(f"train={len(train)} test={len(test)}\n")

# ---- plain Apriori baseline: mine cart + answer, predict a butter for each test cart ----
BUTTERS = {n for n, (b, _) in ITEMS.items() if b == "butter"}
def baseline(level):
    key = (lambda n: ITEMS[n][0]) if level == "base" else (lambda n: n)
    tx = [[key(x) for x in r["cart"] + [r["completion"]]] for r in train]
    te = TransactionEncoder(); df = pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)
    rules = association_rules(apriori(df, min_support=0.05, use_colnames=True), num_itemsets=len(df), metric="confidence", min_threshold=0.5)
    ok = cov = 0; out = []
    for r in test:
        cart = {key(x) for x in r["cart"]}; best = None
        for x in rules.itertuples():
            c = next(iter(x.consequents))
            if len(x.consequents) == 1 and x.antecedents <= cart and (c in BUTTERS or c == "butter"):
                if best is None or x.confidence > best[1]: best = (c, x.confidence)
        pred = best[0] if best else None
        pred_name = "butter" if level == "base" and pred else pred       # base level can only say "butter"
        cov += pred_name is not None; ok += pred_name == r["completion"]
        out.append((r["cart"], pred_name, r["completion"]))
    return ok, cov, out
for level in ("base", "item"):
    ok, cov, out = baseline(level)
    print(f"plain Apriori @ {level} level: correct {ok}/{len(test)}, answered {cov}/{len(test)}")
    for cart, p, a in out: print(f"   {' + '.join(cart):38s} predicted={str(p):15s} expected={a}")
    print()
