
import random
import difflib
import re
import json
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from apriori_core import load_json, mine, predict

# ==========================================
# 1. LOAD BASE MODEL (NO FINE-TUNED ADAPTER)
# ==========================================

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

model, tok = load(MODEL)

print("Loaded BASE Qwen2.5-0.5B-Instruct")
print("No LoRA adapter loaded.")

FIELDS = [
    "category", "labels", "shelf_life_days",
    "storage", "common_uses"
]

INSTRUCTION = "Suggest the best item to add to this cart."

ds2 = {
    d["item_id"]: d
    for d in load_json("ds2_descriptions.json")
}

ID = {d["name"]: i for i, d in ds2.items()}

rules, _ = mine(
    load_json("ds1_train.json"),
    0.004,
    0.3
)

sampler = make_sampler(temp=0)


# ==========================================
# 2. PRODUCT DESCRIPTION
# ==========================================

def describe(i):
    d = ds2[i]
    parts = [i, d["name"]]

    for f in FIELDS:
        v = d[f]
        value = (
            ", ".join(map(str, v))
            if isinstance(v, list) else v
        )
        parts.append(f"{f}: {value}")

    return " | ".join(parts)


# ==========================================
# 3. BUILD SAME PROMPT AS FINE-TUNED MODEL
# ==========================================

def build_prompt(*names, include_confidence=False):
    ids = []

    for n in names:
        if n not in ID:
            raise ValueError(
                f"Unknown product: {n}. "
                f"Closest: {difflib.get_close_matches(n, ID, 3)}"
            )
        ids.append(ID[n])

    hits = predict(rules, ids, k=4)

    cats = list(dict.fromkeys(
        ds2[h[0]]["category"] for h in hits
    ))

    candidates = list(dict.fromkeys(
        i
        for c in cats
        for i, d in ds2.items()
        if d["category"] == c and i not in ids
    ))[:12]

    random.Random(0).shuffle(candidates)

    rule_lines = []

    for h in hits:
        antecedent = " + ".join(
            sorted(ds2[i]["name"] for i in h[3])
        )
        consequent = ds2[h[0]]["name"]

        line = f"{antecedent} => {consequent}"

        if include_confidence:
            line += f" | confidence: {h[1]:.3f}"

        rule_lines.append(line)

    rule_txt = "\n".join(rule_lines) or "(no rule matched)"

    prompt = (
        INSTRUCTION
        + "\n\nCART:\n"
        + "\n".join(describe(i) for i in ids)
        + "\n\nMINED RULES:\n"
        + rule_txt
        + "\n\n"
        + (
            "CANDIDATES:\n"
            + "\n".join(describe(i) for i in candidates)
            + "\n\n"
            if candidates else ""
        )
        + "ANSWER:"
    )

    return prompt, hits, candidates


# ==========================================
# 4. BASE MODEL PREDICTION
# ==========================================

def ask_base(
    *names,
    expected=None,
    include_confidence=False,
    show_prompt=False
):
    prompt, hits, candidates = build_prompt(
        *names,
        include_confidence=include_confidence
    )

    if getattr(tok, "chat_template", None):
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )
    else:
        formatted = prompt

    output = generate(
        model,
        tok,
        prompt=formatted,
        max_tokens=20,
        sampler=sampler
    ).strip()

    match = re.search(r"\bOB_\d+\b", output)
    predicted_id = match.group(0) if match else None

    apriori_id = hits[0][0] if hits else None
    expected_id = ID.get(expected) if expected else None

    canonical_name = (
        ds2[predicted_id]["name"]
        if predicted_id in ds2 else None
    )

    generated_name = (
        output.split("|", 1)[1].strip()
        if "|" in output else None
    )

    id_name_match = (
        generated_name == canonical_name
        if canonical_name and generated_name
        else None
    )

    result = {
        "cart": list(names),
        "expected": expected,
        "expected_id": expected_id,
        "predicted_id": predicted_id,
        "predicted_name": canonical_name,
        "raw_output": output,
        "apriori_top1": apriori_id,
        "model_correct": (
            predicted_id == expected_id
            if expected is not None else None
        ),
        "apriori_correct": (
            apriori_id == expected_id
            if expected is not None else None
        ),
        "expected_in_candidates": (
            expected_id in candidates
            if expected is not None else None
        ),
        "model_in_candidates": predicted_id in candidates,
        "follows_apriori": predicted_id == apriori_id,
        "id_name_match": id_name_match
    }

    print("\n" + "=" * 65)
    print("CART:", " + ".join(names))

    print(
        "APRIORI:",
        ", ".join(
            f"{ds2[h[0]]['name']} ({h[1]:.2f})"
            for h in hits[:3]
        ) or "-- no rule --"
    )

    print("BASE QWEN:", output)
    print("PREDICTED ID:", predicted_id)
    print("CANONICAL NAME:", canonical_name)
    print("IN CANDIDATES:", result["model_in_candidates"])
    print("FOLLOWS APRIORI:", result["follows_apriori"])
    print("ID/NAME MATCH:", id_name_match)

    if expected is not None:
        print("EXPECTED:", expected)
        print("EXPECTED IN CANDIDATES:",
              result["expected_in_candidates"])
        print("MODEL CORRECT:", result["model_correct"])
        print("APRIORI CORRECT:", result["apriori_correct"])

    if show_prompt:
        print("\nFULL PROMPT:\n", prompt)

    print("=" * 65)

    return result


# ==========================================
# 5. SAME EIGHT TEST EXAMPLES
# ==========================================

tests = [
    (("Vegan Burger Patties", "Vegan Hot Dog Buns"), "Vegan Sausages"),
    (("Vegan Burger Patties", "Hot Dog Buns"), "Veggie Sausages"),
    (("Vegan Cheese Slices", "Bananas", "Sourdough"), "Vegan Mayo"),
    (("Gluten-Free Muesli", "Almond Milk"), "Coconut Yogurt"),
    (("Sourdough", "Soy Milk"), "Vegan Butter"),
    (("Corn Flakes", "Lactose-Free Milk", "Basmati Rice"), "Lactose-Free Yogurt"),
    (("Skimmed Milk", "Plain Flour"), "Egg Whites Carton"),
    (("Vegan Bolognese Sauce", "Egg Tagliatelle", "Dishwasher Tablets"), "Vegetarian Hard Cheese"),
]

results = []

for cart, expected in tests:
    result = ask_base(
        *cart,
        expected=expected,
        include_confidence=False
    )
    results.append(result)


# ==========================================
# 6. SUMMARY
# ==========================================

n = len(results)

print("\n========== BASELINE SUMMARY ==========")

print(
    "BASE QWEN HIT@1:",
    sum(r["model_correct"] for r in results),
    "/", n
)

print(
    "APRIORI HIT@1:",
    sum(r["apriori_correct"] for r in results),
    "/", n
)

print(
    "EXPECTED IN CANDIDATES:",
    sum(r["expected_in_candidates"] for r in results),
    "/", n
)

print(
    "BASE QWEN FOLLOWS APRIORI:",
    sum(r["follows_apriori"] for r in results),
    "/", n
)

print(
    "BASE QWEN OUTPUT IN CANDIDATES:",
    sum(r["model_in_candidates"] for r in results),
    "/", n
)

print(
    "ID/NAME MISMATCHES:",
    sum(r["id_name_match"] is False for r in results),
    "/", n
)

with open("base_qwen_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved results to base_qwen_results.json")
