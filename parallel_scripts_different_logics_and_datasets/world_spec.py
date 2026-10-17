"""World spec = DATA ONLY. Add items / families here; the scripts never hard-code a family.

ITEMS: (name, category, labels, shelf_life_days, storage, common_uses)
  labels are retailer-style filter tags:  vegan (vegan badge) | plant (plant-based alternative) | veg (no meat)
  meat | dairy | egg | eggless | meat_free | gluten_free | low_fat | lactose_free | dairy_free
FAMILIES: slots (lists of item names the cart is built from) -> outcome chosen by an ordered rule table:
  (outcome, label, min_items_in_cart_with_label, forbidden_labels)  first rule that fires wins, else `default`.
"""
ITEMS = [
 # milks
 ("Whole Milk", "milk", "dairy veg", 7, "chilled", "tea, coffee, cereal"),
 ("Semi-Skimmed Milk", "milk", "dairy veg low_fat", 7, "chilled", "tea, coffee, cereal"),
 ("Skimmed Milk", "milk", "dairy veg low_fat", 7, "chilled", "tea, coffee, cereal"),
 ("Lactose-Free Milk", "milk", "dairy veg lactose_free", 10, "chilled", "tea, coffee, cereal"),
 ("Oat Milk", "milk", "plant vegan veg dairy_free low_fat", 180, "ambient", "coffee, cereal, baking"),
 ("Soy Milk", "milk", "plant vegan veg dairy_free", 180, "ambient", "coffee, cereal, baking"),
 ("Almond Milk", "milk", "plant vegan veg dairy_free", 120, "ambient", "coffee, smoothies, cereal"),
 ("Coconut Milk", "milk", "plant vegan veg dairy_free", 150, "ambient", "curries, coffee, baking"),
 # breads
 ("White Bread", "bread", "veg", 5, "ambient", "toast, sandwiches"),
 ("Wholemeal Bread", "bread", "veg", 5, "ambient", "toast, sandwiches"),
 ("Bran Loaf", "bread", "veg", 6, "ambient", "toast"),
 ("Sourdough", "bread", "vegan veg", 4, "ambient", "toast, sandwiches"),
 ("Gluten-Free Bread", "bread", "vegan veg gluten_free", 7, "ambient", "toast, sandwiches"),
 ("Rye Bread", "bread", "vegan veg", 6, "ambient", "toast, sandwiches"),
 # butters (outcomes)
 ("Salted Butter", "butter", "dairy veg", 60, "chilled", "spreading, baking"),
 ("Fat-Free Butter", "butter", "dairy veg low_fat", 45, "chilled", "spreading"),
 ("Lactose-Free Butter", "butter", "dairy veg lactose_free", 60, "chilled", "spreading, baking"),
 ("Vegan Butter", "butter", "plant vegan veg dairy_free", 120, "chilled", "spreading, baking"),
 # cereals
 ("Corn Flakes", "cereal", "vegan veg", 300, "ambient", "breakfast"),
 ("Muesli", "cereal", "vegan veg", 200, "ambient", "breakfast"),
 ("Gluten-Free Muesli", "cereal", "vegan veg gluten_free", 200, "ambient", "breakfast"),
 ("Honey Granola", "cereal", "veg", 180, "ambient", "breakfast, snacks"),
 # yogurts (outcomes)
 ("Greek Yogurt", "yogurt", "dairy veg", 14, "chilled", "breakfast, dessert"),
 ("Low Fat Yogurt", "yogurt", "dairy veg low_fat", 14, "chilled", "breakfast"),
 ("Lactose-Free Yogurt", "yogurt", "dairy veg lactose_free", 18, "chilled", "breakfast"),
 ("Coconut Yogurt", "yogurt", "plant vegan veg dairy_free", 21, "chilled", "breakfast, dessert"),
 # pasta
 ("Spaghetti", "pasta", "veg", 700, "ambient", "dinner"),
 ("Penne", "pasta", "veg", 700, "ambient", "dinner"),
 ("Egg Tagliatelle", "pasta", "veg egg", 5, "chilled", "dinner"),
 ("Wholewheat Fusilli", "pasta", "vegan veg", 700, "ambient", "dinner"),
 ("Gluten-Free Penne", "pasta", "vegan veg gluten_free", 700, "ambient", "dinner"),
 # pasta sauces: vegan / veg / meat
 ("Tomato & Basil Sauce", "sauce", "vegan veg", 365, "ambient", "pasta"),
 ("Vegan Pesto", "sauce", "vegan veg", 90, "chilled", "pasta"),
 ("Arrabbiata Sauce", "sauce", "vegan veg", 365, "ambient", "pasta"),
 ("Tomato & Mascarpone Sauce", "sauce", "dairy veg", 365, "ambient", "pasta"),
 ("Basil Pesto with Parmesan", "sauce", "dairy veg", 60, "chilled", "pasta"),
 ("Creamy Carbonara Sauce", "sauce", "dairy meat", 120, "chilled", "pasta"),
 ("Beef Bolognese Sauce", "sauce", "meat", 365, "ambient", "pasta, lasagne"),
 ("Meat-Free Bolognese Sauce", "sauce", "dairy veg meat_free", 365, "ambient", "pasta, lasagne"),
 ("Vegan Bolognese Sauce", "sauce", "plant vegan veg meat_free", 365, "ambient", "pasta, lasagne"),
 # cheeses (outcomes)
 ("Parmesan", "cheese", "dairy", 90, "chilled", "pasta, salads"),
 ("Vegetarian Hard Cheese", "cheese", "dairy veg", 90, "chilled", "pasta, salads"),
 ("Vegan Parmesan", "cheese", "plant vegan veg dairy_free", 180, "ambient", "pasta, salads"),
 # sandwich
 ("Brioche Bun", "bun", "veg egg", 4, "ambient", "burgers, sandwiches"),
 ("Eggless Bun", "bun", "veg eggless", 4, "ambient", "burgers, sandwiches"),
 ("Vegan Wrap", "bun", "vegan veg", 14, "ambient", "wraps, sandwiches"),
 ("Ham Slices", "filling", "meat", 7, "chilled", "sandwiches"),
 ("Cheddar Slices", "filling", "dairy veg", 21, "chilled", "sandwiches, toasties"),
 ("Vegan Cheese Slices", "filling", "plant vegan veg dairy_free", 60, "chilled", "sandwiches, toasties"),
 ("Vegan Ham Slices", "filling", "plant vegan veg meat_free", 30, "chilled", "sandwiches"),
 ("Falafel", "filling", "vegan veg", 10, "chilled", "wraps, sandwiches"),
 # mayo (outcomes)
 ("Mayonnaise", "mayo", "egg veg", 90, "ambient", "sandwiches, salads"),
 ("Eggless Mayo", "mayo", "veg eggless", 90, "ambient", "sandwiches, salads"),
 ("Vegan Mayo", "mayo", "plant vegan veg eggless", 90, "ambient", "sandwiches, salads"),
 # baking
 ("Plain Flour", "flour", "veg", 365, "ambient", "baking"),
 ("Self-Raising Flour", "flour", "veg", 365, "ambient", "baking"),
 ("Gluten-Free Flour", "flour", "vegan veg gluten_free", 365, "ambient", "baking"),
 ("Wholemeal Flour", "flour", "vegan veg", 300, "ambient", "baking"),
 # eggs (outcomes)
 ("Free Range Eggs", "eggs", "egg veg", 21, "chilled", "baking, breakfast"),
 ("Egg Replacer", "eggs", "plant vegan veg eggless", 365, "ambient", "baking"),
 ("Egg Whites Carton", "eggs", "egg veg low_fat", 14, "chilled", "baking, breakfast"),
 # bbq / hot dogs
 ("Hot Dog Buns", "hotdog_bun", "veg", 5, "ambient", "bbq"),
 ("Brioche Hot Dog Buns", "hotdog_bun", "veg egg", 4, "ambient", "bbq"),
 ("Vegan Hot Dog Buns", "hotdog_bun", "vegan veg", 5, "ambient", "bbq"),
 ("Gluten-Free Hot Dog Buns", "hotdog_bun", "veg gluten_free", 6, "ambient", "bbq"),
 ("Ketchup", "condiment", "veg", 365, "ambient", "bbq, fries"),
 ("BBQ Sauce", "condiment", "veg", 365, "ambient", "bbq"),
 ("Honey Mustard", "condiment", "veg", 365, "ambient", "bbq, sandwiches"),
 ("Vegan Garlic Sauce", "condiment", "vegan veg", 180, "ambient", "bbq, wraps"),
 ("Beef Burger Patties", "patties", "meat", 4, "chilled", "bbq"),
 ("Meat-Free Burger Patties", "patties", "veg meat_free", 14, "chilled", "bbq"),
 ("Vegan Burger Patties", "patties", "plant vegan veg meat_free", 14, "chilled", "bbq"),
 # sausages (outcomes)
 ("Pork Sausages", "sausages", "meat", 5, "chilled", "bbq, breakfast"),
 ("Veggie Sausages", "sausages", "veg meat_free", 14, "chilled", "bbq, breakfast"),
 ("Vegan Sausages", "sausages", "plant vegan veg meat_free", 14, "chilled", "bbq, breakfast"),
 # neutral fillers (no dietary labels, never affect an outcome)
 ("Bananas", "filler", "", 6, "ambient", "snack"), ("Apples", "filler", "", 14, "ambient", "snack"),
 ("Kitchen Roll", "filler", "", 999, "ambient", "household"), ("Dishwasher Tablets", "filler", "", 999, "ambient", "household"),
 ("Bin Bags", "filler", "", 999, "ambient", "household"), ("Basmati Rice", "filler", "", 700, "ambient", "dinner"),
]
SPECIAL_LABELS = {"vegan", "plant", "low_fat", "lactose_free", "eggless", "meat_free", "gluten_free"}  # labels of niche items (bought less often)
FILLERS = ["Bananas", "Apples", "Kitchen Roll", "Dishwasher Tablets", "Bin Bags", "Basmati Rice"]

_M = ["Whole Milk", "Semi-Skimmed Milk", "Skimmed Milk", "Lactose-Free Milk", "Oat Milk", "Soy Milk", "Almond Milk", "Coconut Milk"]
_B = ["White Bread", "Wholemeal Bread", "Bran Loaf", "Sourdough", "Gluten-Free Bread", "Rye Bread"]
NO_ANIMAL = ["meat", "dairy", "egg"]

FAMILIES = {
 "bread+milk->butter": {"slots": [_M, _B], "default": "Salted Butter", "rules": [
     ("Vegan Butter", "vegan", 2, []), ("Lactose-Free Butter", "lactose_free", 1, []), ("Fat-Free Butter", "low_fat", 1, [])]},
 "cereal+milk->yogurt": {"slots": [["Corn Flakes", "Muesli", "Gluten-Free Muesli", "Honey Granola"], _M], "default": "Greek Yogurt", "rules": [
     ("Coconut Yogurt", "vegan", 2, []), ("Lactose-Free Yogurt", "lactose_free", 1, []), ("Low Fat Yogurt", "low_fat", 1, [])]},
 "pasta+sauce->cheese": {"slots": [
     ["Spaghetti", "Penne", "Egg Tagliatelle", "Wholewheat Fusilli", "Gluten-Free Penne"],
     ["Tomato & Basil Sauce", "Vegan Pesto", "Arrabbiata Sauce", "Tomato & Mascarpone Sauce", "Basil Pesto with Parmesan",
      "Creamy Carbonara Sauce", "Beef Bolognese Sauce", "Meat-Free Bolognese Sauce", "Vegan Bolognese Sauce"]],
     "default": "Parmesan", "rules": [("Vegan Parmesan", "vegan", 1, NO_ANIMAL), ("Vegetarian Hard Cheese", "meat_free", 1, ["meat"])]},
 "bun+filling->mayo": {"slots": [["Brioche Bun", "Eggless Bun", "White Bread", "Sourdough", "Vegan Wrap", "Rye Bread"],
                                 ["Ham Slices", "Cheddar Slices", "Vegan Cheese Slices", "Vegan Ham Slices", "Falafel"]],
     "default": "Mayonnaise", "rules": [("Vegan Mayo", "vegan", 2, NO_ANIMAL), ("Eggless Mayo", "eggless", 1, [])]},
 "flour+milk->eggs": {"slots": [["Plain Flour", "Self-Raising Flour", "Gluten-Free Flour", "Wholemeal Flour"], _M],
     "default": "Free Range Eggs", "rules": [("Egg Replacer", "vegan", 2, NO_ANIMAL), ("Egg Whites Carton", "low_fat", 1, [])]},
 "bbq->sausages": {"slots": [["Hot Dog Buns", "Brioche Hot Dog Buns", "Vegan Hot Dog Buns", "Gluten-Free Hot Dog Buns"],
                             ["Ketchup", "BBQ Sauce", "Honey Mustard", "Vegan Garlic Sauce", "Beef Burger Patties",
                              "Meat-Free Burger Patties", "Vegan Burger Patties"]],
     "default": "Pork Sausages", "rules": [("Vegan Sausages", "vegan", 2, NO_ANIMAL), ("Veggie Sausages", "meat_free", 1, ["meat"])]},
}
# items that never appear in training carts (cold-start): only DS2 knows them
COLD_START = {"Almond Milk", "Vegan Bolognese Sauce", "Vegan Burger Patties"}
