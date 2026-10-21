"""Pure JEV Choice selection among Apriori top-3 candidates (no alpha or rule scores)."""
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


def choose(client, cart, candidate_ids, catalog, retries=3):
    """Send cart descriptions and exactly the supplied candidates to TypeSafe Choice."""
    from typesafe_sdk import Choice

    # Shuffle options deterministically to avoid always putting Apriori #1 first.
    shuffled = list(candidate_ids)
    random.Random(json.dumps([cart, candidate_ids])).shuffle(shuffled)
    option_to_id = {f"option_{i}": item_id for i, item_id in enumerate(shuffled, 1)}
    state = json.dumps({
        "shopping_cart": [describe(i, catalog) for i in cart],
        "task": "Select the single candidate product that best fits this shopping cart.",
    }, ensure_ascii=False)
    criteria = {
        option: json.dumps(describe(item_id, catalog), ensure_ascii=False)
        for option, item_id in option_to_id.items()
    }
    question = Choice(
        instructions=(
            "Which ONE of the listed candidate products best fits the shopping cart? "
            "Consider the cart products, their descriptions, categories, labels and common uses. "
            "Select only one listed option. Do not assume option order implies quality. "
            "Do not invent other candidates."
        ),
        criteria=criteria,
    )
    for attempt in range(retries):
        try:
            response = client.system_one(state=state, questions={"best_fit": question})
            answer = response.answers["best_fit"]
            selected_option = str(answer.choice)
            if selected_option not in option_to_id:
                raise ValueError(f"JEV returned an unknown option: {selected_option}")
            probabilities = {option_to_id[str(k)]: float(v)
                             for k, v in answer.probabilities.items()
                             if str(k) in option_to_id}
            return option_to_id[selected_option], probabilities, float(answer.confidence)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="ds1_train.json")
    parser.add_argument("--test", default="ds1_test.json")
    parser.add_argument("--ds2", default="ds2_descriptions.json")
    parser.add_argument("--limit", type=int, default=0, help="0 = entire test set")
    parser.add_argument("--min-support", type=float, default=0.004)
    parser.add_argument("--min-conf", type=float, default=0.3)
    parser.add_argument("--output", default="results/jev_cart_choice.csv")
    parser.add_argument("--dry-run", action="store_true", help="No paid JEV calls")
    args = parser.parse_args()

    # .env in the same directory as this script, not dependent on shell cwd.
    try:
        from dotenv import load_dotenv
        load_dotenv(HERE / ".env")
    except ImportError:
        if (HERE / ".env").exists():
            raise SystemExit("Install python-dotenv: pip install python-dotenv")

    train, test = load_json(args.train), load_json(args.test)
    catalog = {item["item_id"]: item for item in load_json(args.ds2)}
    rules, n_itemsets = mine(train, args.min_support, args.min_conf)
    if args.limit:
        test = test[:args.limit]
    print(f"Train={len(train)} Test={len(test)} Frequent itemsets={n_itemsets} Rules={len(rules)}")

    client = None
    if not args.dry_run:
        if not os.getenv("TYPESAFE_API_KEY"):
            raise SystemExit("TYPESAFE_API_KEY not found. Put it in .env next to this script.")
        from typesafe_sdk import TypeSafeClient
        client = TypeSafeClient()

    records = []
    try:
        for index, row in enumerate(test, 1):
            top = predict(rules, row["context"], k=3)
            candidates = [item_id for item_id, _, _, _ in top]
            expected = row["outcome"]  # Used ONLY for evaluation, never sent to JEV.
            baseline = candidates[0] if candidates else None
            selected, probs, confidence, error = None, {}, None, ""
            if candidates and not args.dry_run:
                try:
                    selected, probs, confidence = choose(client, row["context"], candidates, catalog)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            records.append({
                "index": index,
                "cart": json.dumps(row["context"]),
                "candidates": json.dumps(candidates),
                "expected": expected,
                "apriori_prediction": baseline or "",
                "apriori_hit1": int(baseline == expected),
                "apriori_hit3": int(expected in candidates),
                "jev_prediction": selected or "",
                "jev_hit1": int(selected == expected) if not args.dry_run and not error else "",
                "jev_probabilities": json.dumps(probs),
                "jev_confidence": confidence if confidence is not None else "",
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
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)

    n = len(records)
    baseline_hits = sum(r["apriori_hit1"] for r in records)
    top3_hits = sum(r["apriori_hit3"] for r in records)
    failures = sum(bool(r["error"]) for r in records)
    summary = {
        "n": n,
        "apriori_hit1": baseline_hits / n if n else None,
        "apriori_hit3": top3_hits / n if n else None,
        "jev_api_failures": failures,
    }
    if not args.dry_run and not failures:
        jev_hits = sum(r["jev_hit1"] for r in records)
        summary.update({
            "jev_hit1": jev_hits / n if n else None,
            "jev_rescues": sum(r["apriori_hit1"] == 0 and r["jev_hit1"] == 1 for r in records),
            "jev_damages": sum(r["apriori_hit1"] == 1 and r["jev_hit1"] == 0 for r in records),
        })
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
