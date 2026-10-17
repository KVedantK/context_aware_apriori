#!/usr/bin/env python3
"""Mine Apriori rules from training transactions and create Qwen SFT datasets.
No mlxtend dependency required: exact frequent itemset counting for cart size <= 3.
"""
import argparse
import itertools
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load(path):
    with open(path, encoding='utf-8') as f: return json.load(f)

def item_text(item_id, catalog):
    d = catalog[item_id]
    return (f"{item_id} | {d['name']} | category: {d['category']} | "
            f"labels: {', '.join(d.get('labels', [])) or 'none'} | "
            f"storage: {d.get('storage', 'unknown')} | "
            f"shelf life: {d.get('shelf_life_days', 'unknown')} days | "
            f"common uses: {d.get('common_uses', 'unknown')}")

def mine(train, min_support, min_confidence, max_len=3):
    transactions = [set(r['context']) | {r['outcome']} for r in train]
    counts = Counter()
    for t in transactions:
        for k in range(1, min(max_len, len(t)) + 1):
            counts.update(itertools.combinations(sorted(t), k))
    n = len(transactions)
    frequent = {items: cnt for items, cnt in counts.items() if cnt/n >= min_support}
    outcomes = {r['outcome'] for r in train}
    rules = []
    for items, joint_count in frequent.items():
        if len(items) < 2: continue
        for consequent in items:
            if consequent not in outcomes: continue
            antecedent = tuple(i for i in items if i != consequent)
            ante_count = counts[antecedent]
            confidence = joint_count / ante_count
            if confidence + 1e-12 < min_confidence: continue
            rules.append({'antecedent': list(antecedent), 'consequent': consequent,
                          'support': joint_count/n, 'confidence': confidence,
                          'support_count': joint_count, 'antecedent_count': ante_count})
    rules.sort(key=lambda r: (-r['confidence'], -r['support'], r['antecedent'], r['consequent']))
    return rules, len(frequent)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=Path, default=ROOT/'ds_train1.json')
    p.add_argument('--catalog', type=Path, default=ROOT/'ds2_new.json')
    p.add_argument('--outdir', type=Path, default=ROOT/'llm_apriori_dataset')
    p.add_argument('--min-support', type=float, default=.002)
    p.add_argument('--min-confidence', type=float, default=.3)
    p.add_argument('--val-ratio', type=float, default=.1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    train = load(args.train)
    catalog = {d['item_id']: d for d in load(args.catalog)}
    missing = ({i for r in train for i in [*r['context'], r['outcome']]} - set(catalog))
    if missing: raise ValueError(f'IDs absent from catalog: {sorted(missing)}')
    rules, nfreq = mine(train, args.min_support, args.min_confidence)
    args.outdir.mkdir(parents=True, exist_ok=True)
    simple, metadata, messages = [], [], []
    system = ('You are a grocery basket completion assistant. Given product details in a shopping cart, '
              'recommend exactly one complementary product. Return only its item ID and name. '
              'Use product category, dietary labels, and common uses as context.')
    for rule in rules:
        ctx = [item_text(i, catalog) for i in rule['antecedent']]
        out = item_text(rule['consequent'], catalog)
        simple.append({'context': ctx, 'outcome': out})
        metadata.append({**rule, 'context_descriptions': ctx, 'outcome_description': out})
        messages.append({'messages': [
            {'role':'system', 'content': system},
            {'role':'user', 'content': 'Shopping cart:\n' + '\n'.join('- '+x for x in ctx) + '\nRecommend one complementary product.'},
            {'role':'assistant', 'content': f"{rule['consequent']} | {catalog[rule['consequent']]['name']}"}
        ]})
    def dump_json(name, data):
        with open(args.outdir/name, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)
    def dump_jsonl(name, data):
        with open(args.outdir/name, 'w', encoding='utf-8') as f:
            for obj in data: f.write(json.dumps(obj, ensure_ascii=False)+'\n')
    dump_json('apriori_enriched_rules.json', simple)
    dump_json('apriori_rules_with_metrics.json', metadata)
    rng = random.Random(args.seed)
    indices = list(range(len(messages))); rng.shuffle(indices)
    val_n = max(1, round(len(indices)*args.val_ratio)) if len(indices)>1 else 0
    dump_jsonl('qwen_sft_train.jsonl', [messages[i] for i in indices[val_n:]])
    dump_jsonl('qwen_sft_val.jsonl', [messages[i] for i in indices[:val_n]])
    print(f'train transactions={len(train)} | frequent itemsets={nfreq} | outcome rules={len(rules)}')
    print(f'Qwen SFT train={len(indices)-val_n}, val={val_n}')
    print(f'Files written to {args.outdir}')

if __name__ == '__main__': main()
