"""Mine and save the same ID-only Apriori rules used by mine_test_full_set.py.

Usage:
    pip install pandas mlxtend
    python mine_apriori_rules.py
    python mine_apriori_rules.py --min_support 0.002 --min_conf 0.3

Reads ds_train1.json next to this script; never reads the test set or product descriptions.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Export baseline Apriori outcome rules as raw item IDs")
    parser.add_argument("--train", type=Path, default=HERE / "ds_train1.json")
    parser.add_argument("--output", type=Path, default=HERE / "apriori_rules.json")
    parser.add_argument("--min_support", type=float, default=0.002)
    parser.add_argument("--min_conf", type=float, default=0.3)
    parser.add_argument("--max_len", type=int, default=3)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    with args.train.open(encoding="utf-8") as f:
        train = json.load(f)
    transactions = [list(dict.fromkeys(r["context"] + [r["outcome"]])) for r in train]
    outcomes = {r["outcome"] for r in train}

    encoder = TransactionEncoder()
    frame = pd.DataFrame(encoder.fit(transactions).transform(transactions), columns=encoder.columns_)
    frequent = apriori(frame, min_support=args.min_support, use_colnames=True, max_len=args.max_len)
    try:
        mined = association_rules(frequent, num_itemsets=len(frame), metric="confidence", min_threshold=args.min_conf)
    except TypeError:  # Older mlxtend versions
        mined = association_rules(frequent, metric="confidence", min_threshold=args.min_conf)

    mined = mined[mined["consequents"].apply(lambda c: len(c) == 1 and next(iter(c)) in outcomes)]
    mined = mined.sort_values(["confidence", "support"], ascending=[False, False], kind="stable")

    rules = []
    for r in mined.itertuples(index=False):
        rules.append({
            "antecedents": sorted(r.antecedents),
            "consequents": sorted(r.consequents),
            "support": float(r.support),
            "confidence": float(r.confidence),
            "lift": float(r.lift),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)

    print(f"train={len(train)} | frequent itemsets={len(frequent)} | outcome rules={len(rules)}")
    print(f"min_support={args.min_support} min_confidence={args.min_conf} max_len={args.max_len}")
    print(f"Saved: {args.output.resolve()}")
    print("\nTop rules (IDs only):")
    for rule in rules[:args.top]:
        lhs = " + ".join(rule["antecedents"])
        rhs = rule["consequents"][0]
        print(f"  {lhs:32s} => {rhs:9s} support={rule['support']:.3f} confidence={rule['confidence']:.2f} lift={rule['lift']:.2f}")


if __name__ == "__main__":
    main()
