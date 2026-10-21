"""Compare Apriori, alpha-blended Apriori + JEV, and pure JEV Choice.

JEV sees ONLY cart and Apriori candidate descriptions, never Apriori scores/ranks.
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

from apriori_core import load_json, mine, predict

HERE = Path(__file__).resolve().parent


def describe(item_id, catalog):
    item = catalog.get(item_id, {})
    return {
        "id": item_id,
        "name": item.get("name", item_id),
        "category": item.get("category", "unknown"),
        "labels": item.get("labels", []),
        "common_uses": item.get("common_uses", ""),
        "storage": item.get("storage", "unknown"),
    }


def ask_jev(client, cart, candidate_ids, catalog, retries=3):
    """Ask JEV to choose among candidates; no Apriori evidence is supplied."""
    from typesafe_sdk import Choice

    options = list(candidate_ids)
    random.Random(json.dumps([cart, sorted(candidate_ids)])).shuffle(options)
    mapping = {f"option_{i}": item for i, item in enumerate(options, 1)}
    state = json.dumps({"shopping_cart": [describe(i, catalog) for i in cart]}, ensure_ascii=False)
    criteria = {key: json.dumps(describe(item, catalog), ensure_ascii=False)
                for key, item in mapping.items()}
    question = Choice(
        instructions=(
            "Choose the ONE candidate product that best fits this shopping cart as a "
            "likely next/additional purchase. Use only the shopping cart and candidate "
            "product information. Select exactly one listed option; do not infer "
            "quality from option order."
        ),
        criteria=criteria,
    )
    for attempt in range(retries):
        try:
            response = client.system_one(state=state, questions={"best_candidate": question})
            answer = response.answers["best_candidate"]
            selected = str(answer.choice)
            probabilities = {str(k): float(v) for k, v in answer.probabilities.items()}
            if selected not in mapping or set(probabilities) != set(mapping):
                raise ValueError("JEV returned missing or unexpected Choice options")
            return mapping[selected], {mapping[k]: v for k, v in probabilities.items()}, float(answer.confidence)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 8))


def normalized_apriori(top):
    """Normalize Apriori rule confidence across the returned candidates to sum to 1."""
    total = sum(max(0.0, float(conf)) for _, conf, _, _ in top)
    if not top:
        return {}
    if total <= 0:
        return {item: 1 / len(top) for item, _, _, _ in top}
    return {item: max(0.0, float(conf)) / total for item, conf, _, _ in top}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="ds1_train.json")
    parser.add_argument("--test", default="ds1_test.json")
    parser.add_argument("--ds2", default="ds2_descriptions.json")
    parser.add_argument("--limit", type=int, default=0, help="0 = full test set")
    parser.add_argument("--min-support", type=float, default=0.004)
    parser.add_argument("--min-conf", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.5, help="Apriori blend weight in [0,1]")
    parser.add_argument("--output", default="results/jev_rerank.csv")
    parser.add_argument("--dry-run", action="store_true", help="Run Apriori without paid JEV calls")
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")

    try:
        from dotenv import load_dotenv
        load_dotenv(HERE / ".env")
    except ImportError:
        if (HERE / ".env").exists():
            raise SystemExit("Install python-dotenv: pip install python-dotenv")

    train, test = load_json(args.train), load_json(args.test)
    catalog = {item["item_id"]: item for item in load_json(args.ds2)}
    rules, itemset_count = mine(train, args.min_support, args.min_conf)
    if args.limit:
        test = test[:args.limit]
    print(f"Train={len(train)} Test={len(test)} Itemsets={itemset_count} Rules={len(rules)} Alpha={args.alpha}")

    client = None
    if not args.dry_run:
        if not os.getenv("TYPESAFE_API_KEY"):
            raise SystemExit("TYPESAFE_API_KEY missing. Place it in .env next to this script.")
        from typesafe_sdk import TypeSafeClient
        client = TypeSafeClient()

    records = []
    try:
        for index, row in enumerate(test, 1):
            top = predict(rules, row["context"], k=3)
            candidates = [item for item, _, _, _ in top]
            apriori = candidates[0] if candidates else None
            expected = row["outcome"]  # Evaluation ONLY, never supplied to JEV.
            a_scores = normalized_apriori(top)
            jev, hybrid, probs, confidence, error = None, None, {}, None, ""
            if candidates and not args.dry_run:
                try:
                    jev, probs, confidence = ask_jev(client, row["context"], candidates, catalog)
                    hybrid_scores = {item: args.alpha * a_scores[item] + (1 - args.alpha) * probs[item]
                                     for item in candidates}
                    # Candidate order breaks exact ties consistently with Apriori ranking.
                    hybrid = max(candidates, key=lambda item: hybrid_scores[item])
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            records.append({
                "index": index,
                "cart": json.dumps(row["context"]),
                "candidates": json.dumps(candidates),
                "expected": expected,
                "apriori_prediction": apriori or "",
                "apriori_hit1": int(apriori == expected),
                "apriori_hit3": int(expected in candidates),
                "apriori_normalized_scores": json.dumps(a_scores),
                "jev_prediction": jev or "",
                "jev_hit1": int(jev == expected) if jev is not None else "",
                "jev_probabilities": json.dumps(probs),
                "jev_confidence": confidence if confidence is not None else "",
                "hybrid_prediction": hybrid or "",
                "hybrid_hit1": int(hybrid == expected) if hybrid is not None else "",
                "error": error,
            })
            if index % 20 == 0 or index == len(test):
                print(f"Processed {index}/{len(test)}")
    finally:
        if client is not None and hasattr(client, "close"):
            client.close()

    output = Path(args.output)
    if not output.is_absolute():
        output = HERE / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if records:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    n = len(records)
    evaluated = [r for r in records if r["jev_hit1"] != ""]
    failures = sum(bool(r["error"]) for r in records)
    summary = {
        "n": n,
        "alpha": args.alpha,
        "apriori_hit1": sum(r["apriori_hit1"] for r in records) / n if n else None,
        "apriori_hit3": sum(r["apriori_hit3"] for r in records) / n if n else None,
        "jev_api_failures": failures,
        "jev_evaluated": len(evaluated),
        "jev_hit1": sum(r["jev_hit1"] for r in evaluated) / len(evaluated) if evaluated else None,
        "hybrid_hit1": sum(r["hybrid_hit1"] for r in evaluated) / len(evaluated) if evaluated else None,
        "jev_rescues": sum(r["apriori_hit1"] == 0 and r["jev_hit1"] == 1 for r in evaluated),
        "jev_damages": sum(r["apriori_hit1"] == 1 and r["jev_hit1"] == 0 for r in evaluated),
        "hybrid_rescues": sum(r["apriori_hit1"] == 0 and r["hybrid_hit1"] == 1 for r in evaluated),
        "hybrid_damages": sum(r["apriori_hit1"] == 1 and r["hybrid_hit1"] == 0 for r in evaluated),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
