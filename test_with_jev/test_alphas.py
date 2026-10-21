
import numpy as np
import csv
import json


def load_records(csv_path):
    records = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            # Convert JSON strings from the CSV into Python objects
            candidates = json.loads(row["candidates"])
            apriori_scores_dict = json.loads(
                row["apriori_normalized_scores"]
            )
            jev_probabilities_dict = json.loads(
                row["jev_probabilities"]
            )

            # Keep scores in the same order as candidates
            apriori_scores = [
                float(apriori_scores_dict[c])
                for c in candidates
            ]

            jev_probabilities = [
                float(jev_probabilities_dict[c])
                for c in candidates
            ]

            records.append({
                "candidates": candidates,
                "apriori_scores": apriori_scores,
                "jev_probabilities": jev_probabilities,
                "actual": row["expected"]
            })

    return records


def find_best_alpha(records):
    results = []

    for alpha in np.arange(0, 1.01, 0.05):
        correct = 0

        for row in records:
            hybrid_scores = [
                alpha * a + (1 - alpha) * j
                for a, j in zip(
                    row["apriori_scores"],
                    row["jev_probabilities"]
                )
            ]

            best_index = int(np.argmax(hybrid_scores))
            prediction = row["candidates"][best_index]

            if prediction == row["actual"]:
                correct += 1

        accuracy = correct / len(records)
        results.append((round(float(alpha), 2), accuracy))

    print("\nALPHA TUNING RESULTS")
    print("-" * 35)

    for alpha, accuracy in results:
        print(f"alpha={alpha:.2f}, Hit@1={accuracy:.4f}")

    best_alpha, best_accuracy = max(
        results, key=lambda x: x[1]
    )

    print("\nBEST RESULT")
    print(f"Best alpha: {best_alpha:.2f}")
    print(f"Best Hit@1: {best_accuracy:.4f}")
    print(f"Correct predictions: {round(best_accuracy * len(records))}/{len(records)}")

    return best_alpha, best_accuracy


if __name__ == "__main__":
    csv_path = "results/jev_rerank.csv"

    records = load_records(csv_path)

    print(f"Loaded {len(records)} records")

    best_alpha, best_accuracy = find_best_alpha(records)
