"""Traditional Apriori baseline with per-case output for expanded datasets.
Run: python mine_test_full_set_detailed.py [--min_support 0.002 --min_conf 0.3]
"""
import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--min_support', type=float, default=0.002)
parser.add_argument('--min_conf', type=float, default=0.3)
parser.add_argument('--test', default='ds_test1(1).json')
parser.add_argument('--top_rules', type=int, default=15)
parser.add_argument('--show', choices=['all', 'misses', 'none'], default='all')
args = parser.parse_args()

train = json.loads((HERE / 'ds_train1.json').read_text())
test = json.loads((HERE / 'ds_test1.json').read_text())
items = {d['item_id']: d for d in json.loads((HERE / 'ds2_new.json').read_text())}
def nm(item_id):
    return items.get(item_id, {}).get('name', item_id)

transactions = [list(set(r['context'] + [r['outcome']])) for r in train]
outcomes = {r['outcome'] for r in train}
encoder = TransactionEncoder()
frame = pd.DataFrame(encoder.fit(transactions).transform(transactions), columns=encoder.columns_)
frequent = apriori(frame, min_support=args.min_support, use_colnames=True, max_len=5)
try:
    rules = association_rules(frequent, num_itemsets=len(frame), metric='confidence', min_threshold=args.min_conf)
except TypeError:
    rules = association_rules(frequent, metric='confidence', min_threshold=args.min_conf)
rules = rules[rules['consequents'].apply(lambda c: len(c) == 1 and next(iter(c)) in outcomes)]
rules = rules.sort_values(['confidence', 'support'], ascending=False)
mined = [(frozenset(r.antecedents), next(iter(r.consequents)), float(r.confidence), float(r.support)) for r in rules.itertuples()]

print(f'train={len(train)} test={len(test)} | frequent itemsets={len(frequent)} | outcome rules={len(mined)} '
      f'(min_support={args.min_support}, min_confidence={args.min_conf})\n\nTop rules:')
for antecedent, outcome, confidence, support in mined[:args.top_rules]:
    left = ' + '.join(sorted(map(nm, antecedent)))
    print(f'  {left:55s} => {nm(outcome):25s} support={support:.3f} confidence={confidence:.2f}')

def predict(cart, k=3):
    cart = set(cart)
    best = {}
    for antecedent, outcome, confidence, support in mined:
        if antecedent <= cart and (confidence, support) > best.get(outcome, (0, 0)):
            best[outcome] = (confidence, support)
    return sorted(best.items(), key=lambda item: item[1], reverse=True)[:k]

stats = defaultdict(lambda: [0, 0, 0, 0])
noise_categories = {'noise', 'spread', 'fruit'}
for kind in ('seen', 'unseen_combo', 'cold_start'):
    group = [r for r in test if r.get('kind', 'other') == kind]
    if not group:
        continue
    if args.show != 'none':
        print(f'\n=== {kind.upper()} (n={len(group)}) ===')
    for r in group:
        top = predict(r['context'])
        ids = [outcome for outcome, _ in top]
        hit1 = bool(ids) and ids[0] == r['outcome']
        hit3 = r['outcome'] in ids
        for key in ('ALL', r.get('kind', 'other'), r.get('family', 'unknown')):
            values = stats[key]
            values[0] += 1
            values[1] += int(hit1)
            values[2] += int(hit3)
            values[3] += int(bool(ids))
        if args.show == 'all' or (args.show == 'misses' and not hit1):
            cart = [nm(i) for i in r['context'] if items.get(i, {}).get('category') not in noise_categories]
            got = ', '.join(f'{nm(o)} ({score[0]:.2f})' for o, score in top) or '-- no rule fired --'
            status = 'HIT ' if hit1 else ('hit@3' if hit3 else 'MISS')
            print(f'[{status}] {" + ".join(cart)}')
            print(f'       family  : {r.get("family", "unknown")}')
            print(f'       expected: {nm(r["outcome"])}')
            print(f'       got     : {got}')

print('\n=== SUMMARY ===')
print(f'{"group":24s}{"n":>5s}{"hit@1":>10s}{"hit@3":>10s}{"answered":>11s}')
for key, (n, h1, h3, answered) in sorted(stats.items(), key=lambda item: (item[0] != 'ALL', item[0])):
    print(f'{key:24s}{n:5d}{h1:5d}/{n:<4d}{h3:5d}/{n:<4d}{answered:5d}/{n:<4d} '
          f'({h1/n:.1%}, {h3/n:.1%}, {answered/n:.1%})')
