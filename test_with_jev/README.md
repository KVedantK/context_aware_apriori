# Apriori + JEV Choice reranking

Uses the original `apriori_core.py` and uploaded DS1/DS2 datasets. `jev_rerank.py` runs two JEV Choice variants (semantic context only, and context plus Apriori evidence), and computes a hybrid with Apriori confidence.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run without paid API calls

```bash
python jev_rerank.py --dry-run
python mine_test.py
```

## Paid JEV runs

Configure the TypeSafe SDK credential (the SDK documentation uses `TYPESAFE_API_KEY`):

```bash
export TYPESAFE_API_KEY='YOUR_KEY'  # PowerShell: $env:TYPESAFE_API_KEY='YOUR_KEY'
python jev_rerank.py --limit 20 --modes both --max-calls 40
python jev_rerank.py --modes both
```

The SDK `system_one` call uses the service's default model (`jev-latest` in the provided documentation). If your account requires an explicit model setting, configure that in the SDK per your account's API version.

## Outputs

`results/summary.json`: baseline and JEV Hit@1, Hit@3 ceiling, conditional reranking accuracy, rescue/damage counts, and family/kind breakdowns.

`results/predictions.csv` and `.json`: row-level predictions and scores.

`results/jev_cache.json`: persistent cache; rerunning with the same prompt avoids repeated charges. Treat this file as sensitive project data.

## Important evaluation rules

- Test `outcome` and `meta` fields are never included in JEV prompts.
- DS2 product descriptions are used as input features; DS1 training rows are used only to mine Apriori rules.
- `--limit` uses the first N test rows (a smoke test, not an unbiased benchmark).
- If the expected item is not among Apriori Top 3, JEV cannot recover it.
- API failures and call caps are recorded; a JEV overall accuracy is not reported if paid calls remain incomplete.
- JEV Choice confidence is distribution concentration, **not** proven prediction correctness.
- Hybrid weight `--alpha` should be tuned on a separate validation set, not the final test set.
- To compare the semantic and Apriori-aware variants, run `--modes both`.
