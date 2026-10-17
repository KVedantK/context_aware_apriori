"""ONE prompt format, shared by dataset creation, evaluation and ad-hoc inference (so they can never drift apart).
CART (every item + its DS2 description) + CART SUMMARY (label counts across the cart) [+ CANDIDATES: Apriori top-k]."""
import json, random
from collections import Counter
from pathlib import Path
from apriori_core import predict

DEFAULT_FIELDS = ["category", "labels", "shelf_life_days", "storage", "common_uses"]
INSTRUCTION = "Suggest the best item for this cart. Choose one of the candidates."
INSTRUCTION_DIRECT = "Suggest the best item for this cart."

def describe(item, fields=DEFAULT_FIELDS):
    parts = [item["item_id"], item["name"]]
    for f in fields:
        if f != "name" and f in item:
            v = item[f]; parts.append(f"{f}: {', '.join(map(str, v)) if isinstance(v, list) else v}")
    return " | ".join(parts)

def cart_summary(cart_ids, ds2, field="labels"):
    """Counts every label of the catalog across the cart's items (zeros included, so 'no meat / no egg' is explicit).
    Generic: uses whatever list-valued `field` the catalog has; '' if there is none."""
    vocab = sorted({l for d in ds2.values() if isinstance(d.get(field), list) for l in d[field]})
    if not vocab: return ""
    cnt = Counter(l for i in cart_ids if isinstance(ds2[i].get(field), list) for l in ds2[i][field])
    return f"CART SUMMARY: {len(cart_ids)} items | " + " | ".join(f"{l} {cnt[l]}" for l in vocab)

def build_prompt(cart_ids, ds2, rules=None, k=3, fields=DEFAULT_FIELDS, instruction=None, order="shuffle", seed=0,
                 candidates=None, show_candidates=True, cart_context=True, summary_field="labels"):
    """returns (prompt, ranked_candidate_ids). ranked = Apriori order (best first), [] when rules is None.
    show_candidates=False -> no Apriori candidates in the prompt: no mining needed at inference."""
    ranked = list(candidates) if candidates is not None else ([h[0] for h in predict(rules, cart_ids, k=k)] if rules is not None else [])
    shown = list(ranked)
    if order == "shuffle": random.Random(seed).shuffle(shown)       # stops the model learning 'pick the first one'
    prompt = (instruction or (INSTRUCTION if show_candidates else INSTRUCTION_DIRECT)) + "\n\nCART:\n" + \
             "\n".join(describe(ds2[i], fields) for i in cart_ids)
    summary = cart_summary(cart_ids, ds2, summary_field) if cart_context else ""
    if summary: prompt += "\n\n" + summary
    if show_candidates:
        prompt += "\n\nCANDIDATES:\n" + ("\n".join(describe(ds2[i], fields) for i in shown) or "(none)")
    return prompt + "\n\nANSWER:", ranked

def load_prompt_config(path="finetune/prompt_config.json"):
    """The prompt settings the dataset was built with -> pass as build_prompt(..., **cfg) at inference."""
    p = Path(path); p = p if p.exists() else Path(__file__).resolve().parent / path
    return json.load(open(p))
