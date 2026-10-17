"""Evaluate Apriori -> JEV Choice reranking without leaking test labels."""
import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from apriori_core import load_json, mine, predict

HERE = Path(__file__).resolve().parent


def product_text(item_id, catalog):
    p = catalog.get(item_id, {})
    if not p:
        return item_id
    return (f"{p.get('name', item_id)} [id={item_id}]; category={p.get('category', 'unknown')}; "
            f"labels={', '.join(p.get('labels', []))}; uses={p.get('common_uses', '')}; "
            f"storage={p.get('storage', 'unknown')}")


def build_request(row, top, catalog, mode):
    # Never include outcome, meta.expected, meta.family, or meta.combo in model state.
    state = {
        'task': 'Predict the most likely next/additional product for this basket.',
        'basket': [product_text(i, catalog) for i in row['context']],
    }
    criteria = {}
    for rank, (item_id, conf, support, antecedent) in enumerate(top, 1):
        key = f'option_{rank}'
        desc = {'product': product_text(item_id, catalog)}
        if mode == 'apriori':
            desc['association_rule'] = {
                'confidence': round(float(conf), 6),
                'support': round(float(support), 6),
                'matched_antecedent': [product_text(i, catalog) for i in sorted(antecedent)],
                'apriori_rank': rank,
            }
        criteria[key] = desc
    return json.dumps(state, sort_keys=True), criteria


def jev_choice(client, state, criteria, retries=3):
    from typesafe_sdk import Choice
    question = Choice(
        instructions=(
            'Which ONE candidate is the most likely actual next/additional product '
            'given the basket? Select only from the listed options. '
            'Use product compatibility and any supplied association-rule evidence. '
            'Do not assume the first candidate is correct.'
        ),
        criteria=criteria,
    )
    for attempt in range(retries):
        try:
            response = client.system_one(state=state, questions={'next_product': question})
            answer = response.answers['next_product']
            probabilities = {str(k): float(v) for k, v in answer.probabilities.items()}
            choice = str(answer.choice)
            if choice not in criteria or set(probabilities) != set(criteria):
                raise ValueError('Unexpected JEV option identifiers')
            return {'choice': choice, 'probabilities': probabilities,
                    'confidence': float(answer.confidence)}
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt + random.random(), 8))


def cache_key(mode, state, criteria):
    payload = json.dumps({'mode': mode, 'state': state, 'criteria': criteria}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def hybrid_prediction(top, probabilities, alpha):
    # Apriori confidence and JEV probabilities are both in [0,1], but not necessarily calibrated.
    scores = []
    for rank, (_, conf, _, _) in enumerate(top, 1):
        scores.append(alpha * float(conf) + (1 - alpha) * probabilities[f'option_{rank}'])
    return top[max(range(len(top)), key=lambda i: scores[i])][0]


def evaluate(records, alpha, modes):
    n = len(records)
    def rate(num, denom=n):
        return round(num / denom, 4) if denom else None
    baseline1 = sum(r['apriori_hit1'] for r in records)
    baseline3 = sum(r['apriori_hit3'] for r in records)
    summary = {'n': n, 'apriori_hit1': rate(baseline1), 'apriori_hit3': rate(baseline3),
               'coverage': rate(sum(bool(r['candidates']) for r in records))}
    for mode in modes:
        eligible = [r for r in records if r.get(f'{mode}_prediction') is not None]
        # Only report all-set Hit@1 when every row has a prediction or is explicitly unanswerable.
        # Rows with no candidates count as misses; failed API calls do not silently count as misses.
        failures = [r for r in records if r['candidates'] and r.get(f'{mode}_prediction') is None]
        summary[f'{mode}_api_failures'] = len(failures)
        if failures:
            summary[f'{mode}_hit1'] = None
            summary[f'{mode}_hybrid_hit1'] = None
            continue
        summary[f'{mode}_hit1'] = rate(sum(r.get(f'{mode}_prediction') == r['expected'] for r in records))
        summary[f'{mode}_hybrid_hit1'] = rate(sum(r.get(f'{mode}_hybrid_prediction') == r['expected'] for r in records))
        summary[f'{mode}_rescue'] = sum(not r['apriori_hit1'] and r.get(f'{mode}_prediction') == r['expected'] for r in records)
        summary[f'{mode}_damage'] = sum(r['apriori_hit1'] and r.get(f'{mode}_prediction') != r['expected'] for r in records)
        summary[f'{mode}_conditional_accuracy'] = rate(
            sum(r.get(f'{mode}_prediction') == r['expected'] for r in records if r['apriori_hit3']), baseline3)
    groups = {}
    for key in ('family', 'kind'):
        for value in sorted({r.get(key, '') for r in records}):
            subset = [r for r in records if r.get(key, '') == value]
            groups[f'{key}:{value}'] = {'n': len(subset), 'apriori_hit1': rate(sum(r['apriori_hit1'] for r in subset), len(subset)),
                                       'apriori_hit3': rate(sum(r['apriori_hit3'] for r in subset), len(subset))}
            for mode in modes:
                if not any(r['candidates'] and r.get(f'{mode}_prediction') is None for r in subset):
                    groups[f'{key}:{value}'][f'{mode}_hit1'] = rate(sum(r.get(f'{mode}_prediction') == r['expected'] for r in subset), len(subset))
    summary['groups'] = groups
    summary['hybrid_alpha'] = alpha
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', default=str(HERE / 'ds1_train.json'))
    ap.add_argument('--test', default=str(HERE / 'ds1_test.json'))
    ap.add_argument('--ds2', default=str(HERE / 'ds2_descriptions.json'))
    ap.add_argument('--min-support', type=float, default=0.004)
    ap.add_argument('--min-conf', type=float, default=0.3)
    ap.add_argument('--limit', type=int, default=0, help='First N test rows; 0=all')
    ap.add_argument('--alpha', type=float, default=0.5, help='Apriori weight in hybrid')
    ap.add_argument('--modes', choices=['semantic', 'apriori', 'both'], default='both')
    ap.add_argument('--output-dir', default=str(HERE / 'results'))
    ap.add_argument('--dry-run', action='store_true', help='Apriori baseline only, no paid API calls')
    ap.add_argument('--max-calls', type=int, default=0, help='Cap new paid API calls; 0=unlimited')
    args = ap.parse_args()
    if not 0 <= args.alpha <= 1:
        ap.error('--alpha must be between 0 and 1')
    train, test = load_json(args.train), load_json(args.test)
    if args.limit:
        test = test[:args.limit]
    catalog = {p['item_id']: p for p in load_json(args.ds2)}
    rules, n_freq = mine(train, args.min_support, args.min_conf)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / 'jev_cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    modes = ['semantic', 'apriori'] if args.modes == 'both' else [args.modes]
    client = None
    if not args.dry_run:
        if not os.getenv('TYPESAFE_API_KEY'):
            raise SystemExit('Set TYPESAFE_API_KEY, or use --dry-run')
        from typesafe_sdk import TypeSafeClient
        client = TypeSafeClient()
    paid_calls = 0
    records = []
    try:
        for idx, row in enumerate(test):
            top = predict(rules, row['context'], k=3)
            ids = [t[0] for t in top]
            rec = {'index': idx, 'expected': row['outcome'], 'family': row.get('meta', {}).get('family', ''),
                   'kind': row.get('meta', {}).get('kind', ''), 'context': row['context'],
                   'candidates': ids, 'apriori_prediction': ids[0] if ids else None,
                   'apriori_hit1': bool(ids) and ids[0] == row['outcome'],
                   'apriori_hit3': row['outcome'] in ids}
            if top and not args.dry_run:
                for mode in modes:
                    state, criteria = build_request(row, top, catalog, mode)
                    key = cache_key(mode, state, criteria)
                    answer = cache.get(key)
                    if answer is None and (not args.max_calls or paid_calls < args.max_calls):
                        try:
                            answer = jev_choice(client, state, criteria)
                            paid_calls += 1
                            cache[key] = answer
                            cache_path.write_text(json.dumps(cache, indent=2))
                        except Exception as exc:
                            rec[f'{mode}_error'] = str(exc)
                            print(f'Warning: row {idx} {mode}: {exc}', flush=True)
                    if answer is not None:
                        selected_idx = int(answer['choice'].split('_')[1]) - 1
                        rec[f'{mode}_prediction'] = ids[selected_idx]
                        rec[f'{mode}_confidence'] = answer['confidence']
                        rec[f'{mode}_probabilities'] = answer['probabilities']
                        rec[f'{mode}_hybrid_prediction'] = hybrid_prediction(top, answer['probabilities'], args.alpha)
            records.append(rec)
            if (idx + 1) % 25 == 0:
                print(f'Processed {idx+1}/{len(test)} | new paid calls={paid_calls}', flush=True)
    finally:
        if client is not None and hasattr(client, 'close'):
            client.close()
    (out / 'predictions.json').write_text(json.dumps(records, indent=2))
    with (out / 'predictions.csv').open('w', newline='') as f:
        fields = ['index', 'family', 'kind', 'expected', 'context', 'candidates', 'apriori_prediction', 'apriori_hit1', 'apriori_hit3']
        for mode in modes:
            fields += [f'{mode}_prediction', f'{mode}_hybrid_prediction', f'{mode}_confidence', f'{mode}_probabilities', f'{mode}_error']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in records:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    if args.dry_run:
        summary = {'n': len(records), 'frequent_itemsets': n_freq, 'rules': len(rules),
                   'apriori_hit1': sum(r['apriori_hit1'] for r in records) / len(records),
                   'apriori_hit3': sum(r['apriori_hit3'] for r in records) / len(records),
                   'coverage': sum(bool(r['candidates']) for r in records) / len(records)}
    else:
        summary = evaluate(records, args.alpha, modes)
        summary.update({'new_paid_calls': paid_calls, 'frequent_itemsets': n_freq, 'rules': len(rules)})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Files saved to {out}')


if __name__ == '__main__':
    main()
