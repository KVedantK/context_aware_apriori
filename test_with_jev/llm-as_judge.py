
import os
import json
import time
import pandas as pd

from pathlib import Path
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv("./.env")

# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "meta-llama/Llama-3.3-70B-Instruct"

JEV_FILE = "jev_only_results_latest_prompt.csv"
APRIORI_FILE = "apriori_results.csv"

OUTPUT_DIR = Path("judge_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_ROWS = 400
MAX_RETRIES = 3

client = InferenceClient(
    api_key=os.getenv("HF_KEY")
)

# ============================================================
# LLM-AS-A-JUDGE SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an impartial expert evaluator of grocery recommendation systems.

Your task is to evaluate ONE recommended grocery product based on
the items currently in a customer's shopping cart.

Evaluate the recommendation independently, without knowing which
recommendation model generated it or what the dataset's expected
outcome is.

Evaluate using THREE criteria:

1. Contextual Relevance (0-3)
0 = The recommendation is unrelated or clearly inappropriate.
1 = The recommendation has a weak or indirect relationship.
2 = The recommendation is reasonable and relevant to the cart.
3 = The recommendation strongly complements the cart, such as
    completing a meal or pairing naturally with existing products.

2. Dietary Compatibility (0-2)
0 = The recommendation clearly violates an explicitly stated
    dietary restriction or a reliable restriction established
    by the provided data.
1 = Dietary compatibility is uncertain because the available
    information is insufficient or ambiguous.
2 = The recommendation is compatible with available dietary
    information, with no meaningful indication of conflict.

3. Practical Usefulness (0-2)
0 = The recommendation is impractical or not a sensible addition.
1 = The recommendation is plausible but has limited usefulness.
2 = The recommendation is a useful and sensible addition.

IMPORTANT RULES:

- Do not assume the customer is vegan solely because one cart
  item is vegan.
- Do not assume lactose intolerance because a customer buys
  plant-based milk.
- Do not assume allergies or medical restrictions.
- Do not invent ingredients, certifications, or dietary labels.
- Ignore unrelated household products when evaluating food pairings.
- Do not penalize a recommendation because another product might
  also be appropriate.
- Do not reward a recommendation merely because it belongs to
  a broad product category.
- Evaluate the actual relationship between the cart and product.
- Be consistent and conservative when assigning scores.
- Treat uncertain dietary evidence as uncertainty, not a
  confirmed dietary violation.
- Evaluate independently of any expected dataset outcome.
- Base your judgment only on the information provided and
  ordinary grocery knowledge.

Return ONLY a valid JSON object with these exact keys:

{
  "contextual_relevance": 0,
  "dietary_compatibility": 0,
  "practical_usefulness": 0,
  "total_score": 0,
  "dietary_status": "compatible",
  "reason": "Brief explanation of the assigned scores."
}

total_score must equal the sum of the three numerical scores.

dietary_status must be one of:
"compatible", "uncertain", "incompatible".

Use "incompatible" only for a clear conflict with an established
dietary restriction.

Do not include Markdown or any text outside the JSON object.
"""


# ============================================================
# JUDGE ONE RECOMMENDATION
# ============================================================

def judge_one(cart, recommendation):

    user_prompt = f"""
Evaluate the following grocery recommendation.

Shopping cart:
{cart}

Recommended product:
{recommendation}

Return the evaluation as JSON.
"""

    response = client.chat_completion(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0,
        max_tokens=350
    )

    content = response.choices[0].message.content

    result = json.loads(content)

    # Validate scores
    score_ranges = {
        "contextual_relevance": (0, 3),
        "dietary_compatibility": (0, 2),
        "practical_usefulness": (0, 2)
    }

    for field, (minimum, maximum) in score_ranges.items():
        value = result[field]

        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"Invalid {field}: {value}")

    if result["dietary_status"] not in {
        "compatible", "uncertain", "incompatible"
    }:
        raise ValueError("Invalid dietary_status")

    if not isinstance(result["reason"], str):
        raise ValueError("Invalid reason")

    # Calculate total ourselves
    result["total_score"] = (
        result["contextual_relevance"]
        + result["dietary_compatibility"]
        + result["practical_usefulness"]
    )

    return result


# ============================================================
# EVALUATE ONE CSV
# ============================================================

def evaluate_file(input_file, model_name):

    df = pd.read_csv(input_file)

    if not {"cart", "outcome"}.issubset(df.columns):
        raise ValueError(
            f"{input_file} must contain cart and outcome columns"
        )

    df = df.head(MAX_ROWS)

    output_file = OUTPUT_DIR / f"{model_name}_judged.csv"

    # Resume completed evaluations
    if output_file.exists():
        results = pd.read_csv(output_file).to_dict("records")
        completed_ids = {
            int(row["row_id"])
            for row in results
        }
    else:
        results = []
        completed_ids = set()

    print(f"\nEvaluating {model_name}: {len(df)} rows")

    for index, row in df.iterrows():

        row_id = index + 1

        if row_id in completed_ids:
            continue

        cart = row["cart"]
        recommendation = row["outcome"]

        # Keep missing Apriori predictions visible
        if pd.isna(recommendation) or not str(recommendation).strip():

            results.append({
                "row_id": row_id,
                "cart": cart,
                "outcome": None,
                "contextual_relevance": None,
                "dietary_compatibility": None,
                "practical_usefulness": None,
                "total_score": None,
                "dietary_status": "no_recommendation",
                "reason": "Apriori produced no recommendation."
            })

            completed_ids.add(row_id)

            pd.DataFrame(results).to_csv(
                output_file,
                index=False
            )

            continue

        for attempt in range(MAX_RETRIES):

            try:
                score = judge_one(
                    str(cart),
                    str(recommendation)
                )

                results.append({
                    "row_id": row_id,
                    "cart": cart,
                    "outcome": recommendation,
                    **score
                })

                completed_ids.add(row_id)

                pd.DataFrame(results).to_csv(
                    output_file,
                    index=False
                )

                print(
                    f"{model_name} | "
                    f"{row_id}/{len(df)} | "
                    f"Score: {score['total_score']}/7"
                )

                break

            except Exception as error:

                print(
                    f"Error on {model_name} row {row_id}, "
                    f"attempt {attempt + 1}: {error}"
                )

                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** (attempt + 1))

        else:
            print(
                f"FAILED: {model_name} row {row_id}. "
                "Will retry on next run."
            )

    return pd.read_csv(output_file)


# ============================================================
# RUN BOTH EVALUATIONS
# ============================================================

jev_scores = evaluate_file(
    JEV_FILE,
    "jev"
)

apriori_scores = evaluate_file(
    APRIORI_FILE,
    "apriori"
)


# ============================================================
# SUMMARY
# ============================================================

def summarize(df, model_name):

    valid = df.dropna(subset=["total_score"])

    return {
        "model": model_name,
        "evaluated": len(valid),
        "average_relevance": valid["contextual_relevance"].mean(),
        "average_dietary": valid["dietary_compatibility"].mean(),
        "average_usefulness": valid["practical_usefulness"].mean(),
        "average_total": valid["total_score"].mean(),
        "high_quality_rate": (
            valid["total_score"] >= 6
        ).mean(),
        "dietary_conflict_rate": (
            valid["dietary_status"] == "incompatible"
        ).mean()
    }


summary = pd.DataFrame([
    summarize(jev_scores, "JEV"),
    summarize(apriori_scores, "Apriori")
])

summary.to_csv(
    OUTPUT_DIR / "comparison_summary.csv",
    index=False
)

print("\n========== FINAL COMPARISON ==========")
print(summary.round(3).to_string(index=False))


# ============================================================
# PAIRED COMPARISON: SAME TEST CARTS
# ============================================================

paired = jev_scores[
    ["row_id", "total_score"]
].merge(
    apriori_scores[["row_id", "total_score"]],
    on="row_id",
    suffixes=("_jev", "_apriori")
)

paired = paired.dropna()

if len(paired) > 0:

    paired["difference"] = (
        paired["total_score_jev"]
        - paired["total_score_apriori"]
    )

    print("\n========== PAIRED COMPARISON ==========")
    print("Comparable carts:", len(paired))

    print(
        "JEV wins:",
        (paired["difference"] > 0).sum()
    )

    print(
        "Apriori wins:",
        (paired["difference"] < 0).sum()
    )

    print(
        "Ties:",
        (paired["difference"] == 0).sum()
    )

    print(
        "Average score difference (JEV - Apriori):",
        round(paired["difference"].mean(), 3)
    )

    paired.to_csv(
        OUTPUT_DIR / "paired_comparison.csv",
        index=False
    )
