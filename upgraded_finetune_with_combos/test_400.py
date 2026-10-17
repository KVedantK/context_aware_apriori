"""Compare Apriori, base Qwen and MLX LoRA Qwen on identical held-out prompts.

Run from the project root:
    python compare_400.py --test finetune/finetune_test.jsonl
    python compare_400.py --test finetune_test_with_candidates.jsonl
    python compare_400.py --test finetune/finetune_test.jsonl --limit 20

No training occurs. Requires: pip install mlx-lm
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--test', default='finetune/finetune_test.jsonl')
parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
parser.add_argument('--adapter', default='out_mlx')
parser.add_argument('--ds2', default='ds2_descriptions.json')
parser.add_argument('--output-dir', default='comparison_400')
parser.add_argument('--limit', type=int, default=0, help='0 means all examples')
parser.add_argument('--max-tokens', type=int, default=24)
parser.add_argument('--no-chat-template', action='store_true')
args = parser.parse_args()


def resolve(path):
    p = Path(path)
    return p if p.exists() else HERE / p


def read_jsonl(path):
    with open(resolve(path), encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


rows = read_jsonl(args.test)
if args.limit:
    rows = rows[:args.limit]
if not rows:
    raise ValueError('Test file is empty')
for i, row in enumerate(rows):
    if 'prompt' not in row or 'outcome_id' not in row.get('meta', {}):
        raise ValueError(f'Row {i} lacks prompt or meta.outcome_id')

products = {d['item_id']: d for d in json.load(open(resolve(args.ds2), encoding='utf-8'))}
output_dir = resolve(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Preserve test prompts exactly: never rebuild candidates or leak labels.
results = []
for i, row in enumerate(rows):
    meta = row['meta']
    expected = meta['outcome_id']
    candidates = meta.get('candidates', [])
    top1 = meta.get('apriori_top1')
    results.append({
        'index': i,
        'family': meta.get('family', ''),
        'kind': meta.get('kind', ''),
        'shows_candidates': 'CANDIDATES:' in row['prompt'],
        'expected_id': expected,
        'expected_name': products.get(expected, {}).get('name'),
        'candidate_count': len(candidates),
        'expected_in_candidates': expected in candidates,
        'apriori_top1': top1,
        'apriori_correct': top1 == expected,
        'candidates': candidates,
        'prompt': row['prompt'],
    })

sampler = make_sampler(temp=0)


def run_model(label, adapter_path=None):
    print(f'\nLoading {label}...', flush=True)
    if adapter_path is None:
        model, tok = load(args.model)
    else:
        model, tok = load(args.model, adapter_path=str(resolve(adapter_path)))

    for i, (row, result) in enumerate(zip(rows, results), 1):
        prompt = row['prompt']
        if not args.no_chat_template and getattr(tok, 'chat_template', None):
            prompt = tok.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        raw = generate(model, tok, prompt=prompt,
                       max_tokens=args.max_tokens, sampler=sampler).strip()
        # Match an item ID at the start, allowing leading whitespace/markdown.
        match = re.match(r'^\s*(?:\*\*)?(OB_\d+)\b', raw)
        predicted = match.group(1) if match else None
        canonical = products.get(predicted, {}).get('name')
        # Name comparison only uses the second pipe-delimited field.
        parts = raw.split('|')
        written_name = parts[1].strip().strip('* ') if len(parts) > 1 else None
        name_match = (written_name == canonical) if written_name and canonical else None
        prefix = 'base' if adapter_path is None else 'finetuned'
        result.update({
            f'{prefix}_raw': raw,
            f'{prefix}_id': predicted,
            f'{prefix}_name': canonical,
            f'{prefix}_correct': predicted == result['expected_id'],
            f'{prefix}_valid_id': predicted in products,
            f'{prefix}_in_candidates': predicted in row['meta'].get('candidates', []),
            f'{prefix}_follows_apriori': predicted is not None and predicted == result['apriori_top1'],
            f'{prefix}_id_name_match': name_match,
        })
        if i % 25 == 0 or i == len(rows):
            print(f'  {label}: {i}/{len(rows)}', flush=True)
    del model, tok


run_model('Base Qwen')
run_model('Fine-tuned Qwen', args.adapter)


def group_metrics(items):
    n = len(items)
    def count(key):
        return sum(bool(r[key]) for r in items)
    return {
        'n': n,
        'apriori_hit1': count('apriori_correct') / n,
        'base_hit1': count('base_correct') / n,
        'finetuned_hit1': count('finetuned_correct') / n,
        'candidate_recall': count('expected_in_candidates') / n,
        'base_valid_id': count('base_valid_id') / n,
        'finetuned_valid_id': count('finetuned_valid_id') / n,
        'base_in_candidates': count('base_in_candidates') / n,
        'finetuned_in_candidates': count('finetuned_in_candidates') / n,
        'base_follows_apriori': count('base_follows_apriori') / n,
        'finetuned_follows_apriori': count('finetuned_follows_apriori') / n,
        'finetuned_fixed_base': sum(r['finetuned_correct'] and not r['base_correct'] for r in items),
        'finetuned_broke_base': sum(r['base_correct'] and not r['finetuned_correct'] for r in items),
    }


groups = {'ALL': results}
for key in ('family', 'kind'):
    values = sorted(set(str(r[key]) for r in results))
    for value in values:
        groups[f'{key}: {value}'] = [r for r in results if str(r[key]) == value]
groups['expected in candidates: True'] = [r for r in results if r['expected_in_candidates']]
groups['expected in candidates: False'] = [r for r in results if not r['expected_in_candidates']]
summary = {name: group_metrics(items) for name, items in groups.items() if items}

print('\n' + '=' * 84)
print(f'{"GROUP":36s} {"N":>5s} {"APRIORI":>10s} {"BASE":>10s} {"FINE-TUNED":>12s}')
for name, s in summary.items():
    print(f'{name[:36]:36s} {s["n"]:5d} {s["apriori_hit1"]:10.1%} {s["base_hit1"]:10.1%} {s["finetuned_hit1"]:12.1%}')
print('=' * 84)
all_stats = summary['ALL']
print('Expected answer in stored Apriori candidates:', f'{all_stats["candidate_recall"]:.1%}')
print('Fine-tuned fixes of base mistakes:', all_stats['finetuned_fixed_base'])
print('Fine-tuned regressions vs base:', all_stats['finetuned_broke_base'])
print('Base valid IDs:', f'{all_stats["base_valid_id"]:.1%}')
print('Fine-tuned valid IDs:', f'{all_stats["finetuned_valid_id"]:.1%}')
print('NOTE: Candidate recall is a ceiling ONLY for a strictly candidate-constrained reranker.')

with open(output_dir / 'predictions.jsonl', 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
with open(output_dir / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['group'] + list(next(iter(summary.values())).keys()))
    writer.writeheader()
    for group, stats in summary.items():
        writer.writerow({'group': group, **stats})
print(f'\nSaved predictions.jsonl, summary.json and summary.csv to {output_dir}')
