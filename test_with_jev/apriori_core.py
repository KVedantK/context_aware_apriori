"""Shared helpers: traditional Apriori (support + confidence) on DS1 and rule matching."""
import json, warnings
from pathlib import Path
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

def load_json(name):
    p = Path(name)
    return json.load(open(p if p.is_absolute() or p.exists() else HERE / name))

def mine(rows, min_support=0.004, min_conf=0.3, max_len=4):
    """Transactions = context + outcome. Keeps single-consequent rules whose consequent is an outcome item."""
    tx = [r["context"] + [r["outcome"]] for r in rows]
    outcomes = {r["outcome"] for r in rows}
    te = TransactionEncoder(); df = pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)
    freq = apriori(df, min_support=min_support, use_colnames=True, max_len=max_len)
    try:    rules = association_rules(freq, num_itemsets=len(df), metric="confidence", min_threshold=min_conf)
    except TypeError: rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    out = []
    for r in rules.itertuples():
        c = next(iter(r.consequents))
        if len(r.consequents) == 1 and c in outcomes and not (r.antecedents & outcomes):
            out.append((frozenset(r.antecedents), c, r.support, r.confidence, r.lift))
    out.sort(key=lambda x: (-x[3], -x[2]))
    return out, len(freq)

def predict(rules, cart_ids, k=3):
    """Rules whose antecedent is inside the cart; best rule per outcome; top-k by confidence.
    returns [(outcome_id, confidence, support, antecedent)]"""
    cart, best = set(cart_ids), {}
    for ant, c, sup, conf, _ in rules:
        if c not in cart and ant <= cart and (c not in best or (conf, sup) > (best[c][0], best[c][1])):
            best[c] = (conf, sup, ant)
    return [(c, *v) for c, v in sorted(best.items(), key=lambda x: (-x[1][0], -x[1][1]))[:k]]
