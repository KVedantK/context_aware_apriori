"""Traditional Apriori (support + confidence) on DS1, then test on mixed-and-matched contexts.
Needs ds1_base_dataset.json + ds2_descriptions.json next to this file.   Run: python mine_and_test.py"""
import json, warnings
from pathlib import Path
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
MIN_SUPPORT, MIN_CONF = 0.02, 0.3

ds1 = json.load(open(HERE / "ds_train1.json"))
ds2 = {d["item_id"]: d for d in json.load(open(HERE / "ds2_new.json"))}
ID = {d["name"]: i for i, d in ds2.items()}
nm = lambda i: ds2[i]["name"]

# ---------- 1. traditional mining on DS1 (ids only; DS2 is NOT used here) ----------
tx = [r["context"] + [r["outcome"]] for r in ds1]
outcomes = {r["outcome"] for r in ds1}
te = TransactionEncoder(); df = pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)
freq = apriori(df, min_support=MIN_SUPPORT, use_colnames=True)
try:    rules = association_rules(freq, num_itemsets=len(df), metric="confidence", min_threshold=MIN_CONF)
except TypeError: rules = association_rules(freq, metric="confidence", min_threshold=MIN_CONF)
rules = rules[rules["consequents"].apply(lambda c: len(c) == 1 and next(iter(c)) in outcomes)]
rules = rules.sort_values(["confidence", "support"], ascending=False)
print(f"DS1 rows={len(ds1)} | frequent itemsets={len(freq)} | outcome rules={len(rules)} "
      f"(min_support={MIN_SUPPORT}, min_confidence={MIN_CONF})\n\nTop rules:")
for r in rules.head(15).itertuples():
    print(f"  {' + '.join(nm(i) for i in sorted(r.antecedents, key=nm)):45s} => {nm(next(iter(r.consequents))):16s}"
          f" support={r.support:.3f} confidence={r.confidence:.2f}")

def predict(cart_ids, k=3):
    """Every rule whose antecedent is inside the cart; best confidence per outcome; top k."""
    best = {}
    for r in rules.itertuples():
        if r.antecedents <= set(cart_ids):
            o = next(iter(r.consequents))
            if o not in best or (r.confidence, r.support) > best[o][:2]:
                best[o] = (r.confidence, r.support)
    return sorted(best.items(), key=lambda x: x[1], reverse=True)[:k]

# ---------- 2. tests ----------
# expected outcome comes from DS2 labels: all cart items vegan -> vegan version; any low_fat -> low-fat version; else default
SEEN = "SEEN in DS1 (control)"; MIX = "MIXED contexts (unseen combos)"
TESTS = [
 (SEEN, ["Whole Milk", "White Bread"],               "Salted Butter",  "plain rule"),
 (SEEN, ["Oat Milk", "Sourdough"],                   "Vegan Butter",   "specific rule"),
 (SEEN, ["Spaghetti", "Vegan Pesto"],                "Vegan Parmesan", "specific rule"),
 (MIX,  ["Oat Milk", "White Bread"],                 "Fat-Free Butter","oat milk is low_fat, white bread isn't vegan"),
 (MIX,  ["Oat Milk", "Wholemeal Bread"],             "Fat-Free Butter","oat milk is low_fat, wholemeal isn't vegan"),
 (MIX,  ["Soy Milk", "White Bread"],                 "Salted Butter",  "only one vegan signal"),
 (MIX,  ["Skimmed Milk", "Sourdough"],               "Fat-Free Butter","skimmed milk is low_fat"),
 (MIX,  ["Whole Milk", "Sourdough"],                 "Salted Butter",  "dairy milk, one vegan signal"),
 (MIX,  ["Almond Milk", "Sourdough"],                "Vegan Butter",   "almond milk never in DS1; both vegan"),
 (MIX,  ["Almond Milk", "Gluten-Free Bread"],        "Vegan Butter",   "almond milk never in DS1; both vegan"),
 (MIX,  ["Corn Flakes", "Oat Milk"],                 "Coconut Yogurt", "both vegan"),
 (MIX,  ["Corn Flakes", "Soy Milk"],                 "Coconut Yogurt", "both vegan"),
 (MIX,  ["Granola", "Skimmed Milk"],                 "Greek Yogurt",   "dairy milk"),
 (MIX,  ["Gluten-Free Penne", "Tomato & Mascarpone Sauce"], "Parmesan","sauce is dairy"),
 (MIX,  ["Gluten-Free Penne", "Creamy Carbonara Sauce"],    "Parmesan","sauce is dairy + meat"),
]

stats = {}
for group in (SEEN, MIX):
    print(f"\n=== {group} ===")
    for g, cart, exp, why in TESTS:
        if g != group: continue
        top = predict([ID[x] for x in cart])
        names = [nm(o) for o, _ in top]
        h1, h3 = bool(names) and names[0] == exp, exp in names
        s = stats.setdefault(group, [0, 0, 0, 0]); s[0] += 1; s[1] += h1; s[2] += h3; s[3] += bool(names)
        got = ", ".join(f"{nm(o)} ({c:.2f})" for o, (c, _) in top) or "-- no rule --"
        print(f"[{'HIT ' if h1 else ('hit@3' if h3 else 'MISS')}] {' + '.join(cart)}\n        expected: {exp}  ({why})\n        got     : {got}")

print("\n=== SUMMARY ===")
for g, (n, h1, h3, cov) in stats.items():
    print(f"{g:34s} n={n:2d}  hit@1={h1}/{n}  hit@3={h3}/{n}  answered={cov}/{n}")