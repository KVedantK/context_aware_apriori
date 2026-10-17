"""Builds DS1 (base transactions, ids only) and DS2 (item descriptions). Run: python build_datasets.py"""
import json, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
rng = random.Random(42)

# ---------------- DS2: descriptions ----------------
# (name, category, labels, shelf_life_days, storage, common_uses)
RAW = [
 ("Whole Milk",        "milk",   ["dairy", "veg"],                                      7,   "chilled", "tea, coffee, cereal"),
 ("Skimmed Milk",      "milk",   ["dairy", "veg", "low_fat"],                           7,   "chilled", "tea, coffee, cereal"),
 ("Oat Milk",          "milk",   ["plant", "vegan", "veg", "dairy_free", "low_fat"],    180, "ambient", "coffee, cereal, baking"),
 ("Soy Milk",          "milk",   ["plant", "vegan", "veg", "dairy_free"],               180, "ambient", "coffee, cereal, baking"),
 ("Almond Milk",       "milk",   ["plant", "vegan", "veg", "dairy_free"],               120, "ambient", "coffee, smoothies"),
 ("White Bread",       "bread",  ["veg"],                                               5,   "ambient", "toast, sandwiches"),
 ("Wholemeal Bread",   "bread",  ["veg"],                                               5,   "ambient", "toast, sandwiches"),
 ("Sourdough",         "bread",  ["plant", "vegan", "veg"],                             4,   "ambient", "toast, sandwiches"),
 ("Gluten-Free Bread", "bread",  ["plant", "vegan", "veg", "gluten_free"],              7,   "ambient", "toast, sandwiches"),
 ("Salted Butter",     "butter", ["dairy", "veg"],                                      60,  "chilled", "spreading, baking"),
 ("Fat-Free Butter",   "butter", ["dairy", "veg", "low_fat"],                           45,  "chilled", "spreading"),
 ("Vegan Butter",      "butter", ["plant", "vegan", "veg", "dairy_free"],               120, "chilled", "spreading, baking"),
 ("Corn Flakes",       "cereal", ["plant", "vegan", "veg"],                             300, "ambient", "breakfast"),
 ("Granola",           "cereal", ["plant", "vegan", "veg"],                             200, "ambient", "breakfast, snacks"),
 ("Greek Yogurt",      "yogurt", ["dairy", "veg"],                                      14,  "chilled", "breakfast, dessert"),
 ("Coconut Yogurt",    "yogurt", ["plant", "vegan", "veg", "dairy_free"],               21,  "chilled", "breakfast, dessert"),
 ("Spaghetti",         "pasta",  ["plant", "vegan", "veg"],                             700, "ambient", "dinner"),
 ("Gluten-Free Penne", "pasta",  ["plant", "vegan", "veg", "gluten_free"],              700, "ambient", "dinner"),
 ("Tomato & Mascarpone Sauce", "sauce", ["dairy", "veg"],                               365, "ambient", "pasta"),
 ("Creamy Carbonara Sauce",    "sauce", ["dairy", "meat"],                              120, "chilled", "pasta"),
 ("Vegan Pesto",       "sauce",  ["plant", "vegan", "veg"],                             90,  "chilled", "pasta"),
 ("Parmesan",          "cheese", ["dairy"],                                             90,  "chilled", "pasta, salads"),
 ("Vegan Parmesan",    "cheese", ["plant", "vegan", "veg", "dairy_free"],               180, "ambient", "pasta, salads"),
 ("Eggs",              "eggs",   ["veg"],                                               21,  "chilled", "breakfast, baking"),
 ("Strawberry Jam",    "spread", ["plant", "vegan", "veg"],                             365, "ambient", "toast"),
 ("Bananas",           "fruit",  ["plant", "vegan", "veg"],                             6,   "ambient", "snack, cereal"),
]
ds2 = [{"item_id": f"OB_{1001 + k}", "name": n, "category": c, "labels": l,
        "shelf_life_days": s, "storage": st, "common_uses": u} for k, (n, c, l, s, st, u) in enumerate(RAW)]
ID = {d["name"]: d["item_id"] for d in ds2}

# ---------------- DS1: base dataset (ids only) ----------------
# (context names, outcome name, copies, is_general)
FAMILIES = [
 # butter
 (["Whole Milk", "White Bread"],      "Salted Butter",  40, True),
 (["Whole Milk", "Wholemeal Bread"],  "Salted Butter",  20, True),
 (["Oat Milk", "Sourdough"],          "Vegan Butter",   15, False),
 (["Soy Milk", "Sourdough"],          "Vegan Butter",   12, False),
 (["Oat Milk", "Gluten-Free Bread"],  "Vegan Butter",    8, False),
 (["Skimmed Milk", "Wholemeal Bread"],"Fat-Free Butter",14, False),
 # yogurt
 (["Corn Flakes", "Whole Milk"],      "Greek Yogurt",   25, True),
 (["Granola", "Whole Milk"],          "Greek Yogurt",   10, True),
 (["Granola", "Soy Milk"],            "Coconut Yogurt", 10, False),
 # cheese
 (["Spaghetti", "Tomato & Mascarpone Sauce"], "Parmesan",       25, True),
 (["Spaghetti", "Creamy Carbonara Sauce"],    "Parmesan",       15, True),
 (["Spaghetti", "Vegan Pesto"],               "Vegan Parmesan", 10, False),
 (["Gluten-Free Penne", "Vegan Pesto"],       "Vegan Parmesan",  8, False),
]
GENERAL_OUT = {"Vegan Butter": "Salted Butter", "Fat-Free Butter": "Salted Butter",
               "Coconut Yogurt": "Greek Yogurt", "Vegan Parmesan": "Parmesan"}
FILLERS = ["Eggs", "Strawberry Jam", "Bananas"]

ds1 = []
for ctx, out, n, general in FAMILIES:
    for _ in range(n):
        o = out
        if not general and rng.random() < 0.12:      # noise: shopper occasionally buys the regular version
            o = GENERAL_OUT[out]
        c = list(ctx)
        if rng.random() < 0.25: c.append(rng.choice(FILLERS))   # noise: extra item in the cart
        ds1.append({"context": [ID[x] for x in c], "outcome": ID[o]})
rng.shuffle(ds1)

json.dump(ds1, open(HERE / "ds1.json", "w"), indent=1)
json.dump(ds2, open(HERE / "ds2.json", "w"), indent=1)
print(f"DS1: {len(ds1)} rows | DS2: {len(ds2)} items")